# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
from copy import deepcopy
from tempfile import NamedTemporaryFile

import numpy as np
from datasets import load_dataset, concatenate_datasets

from sklearn.metrics import accuracy_score, confusion_matrix
from evaluator import MoreEvaluator, prepro

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    AdamW,
    get_constant_schedule,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from unify_trainer import DataCollatorForUnify

from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration
from scipy.special import softmax

ENTAILMENT_CLASSES = ['yes', 'no', 'unknown']

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    tokenizer_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the customized tokenizer data."}
    )
    train_qa_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training qa data."}
    )
    validation_qa_seen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation seen qa data."}
    )
    validation_qa_unseen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation unseen qa data."}
    )
    test_qa_seen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test seen qa data."}
    )
    test_qa_unseen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test unseen qa data."}
    )
    train_retrieval_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training retrieval tfidf data."}
    )
    validation_retrieval_seen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation tfidf data."}
    )
    validation_retrieval_unseen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation tfidf data."}
    )
    test_retrieval_seen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test tfidf data."}
    )
    test_retrieval_unseen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test tfidf data."}
    )
    snippet_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the snippet data."}
    )
    top_k_snippets: Optional[int] = field(
        default=100, metadata={"help": "Use top-k retrieved snippets for training"}
    )
    tokenized_file: Optional[str] = field(
        default=None, metadata={"help": "tokenized sharc data"}
    )
    tree_mapping_file: Optional[str] = field(
        default=None, metadata={"help": "map tree_id to its own reasoning structure such as follow_up questions / clauses"}
    )
    debug_sharc: bool = field(
        default=False,
        metadata={"help": "debug model, load less data"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    max_target_length: Optional[int] = field(
        default=30,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": "number of beams when generate with beam search."
        },
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    data_files = {
        "train": data_args.train_qa_file,
        "validation_seen": data_args.validation_qa_seen_file,
        "validation_unseen": data_args.validation_qa_unseen_file,
        "test_seen": data_args.test_qa_seen_file,
        "test_unseen": data_args.test_qa_unseen_file,
                  }


    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_qa_file.endswith(".csv"):
        datasets = load_dataset("csv", data_files=data_files)
    else:
        datasets = load_dataset("json", data_files=data_files)
    
    def _get_label(example):
        example['label'] = example['answer'].lower()
        return example
    datasets = datasets.map(_get_label)

    def _load_json(path):
        with open(path) as f:
            data = json.load(f)
        return data

    id2snippet = _load_json(data_args.snippet_file)

    snippet2id = {v:k for k, v in id2snippet.items()}
    
    retrieval_snippet_id_train = _load_json(data_args.train_retrieval_file)
    
    retrieval_snippet_id_dev_seen = _load_json(data_args.validation_retrieval_seen_file)
    retrieval_snippet_id_dev_unseen = _load_json(data_args.validation_retrieval_unseen_file)
    retrieval_snippet_id_test_seen = _load_json(data_args.test_retrieval_seen_file)
    retrieval_snippet_id_test_unseen = _load_json(data_args.test_retrieval_unseen_file)
    assert len(retrieval_snippet_id_train) == len(datasets['train']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_dev_seen) == len(datasets['validation_seen']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_dev_unseen) == len(datasets['validation_unseen']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_test_seen) == len(datasets['test_seen']), 'Examples mismatch!'
    assert len(retrieval_snippet_id_test_unseen) == len(datasets['test_unseen']), 'Examples mismatch!'



    tokenized_sharc_data = _load_json(data_args.tokenized_file)
    
    tree_mapping_data = _load_json(data_args.tree_mapping_file)
    
    snippetid2snippetparsed = {}

    for v in tree_mapping_data.values():
        if snippet2id[v['snippet']] not in snippetid2snippetparsed:
            snippetid2snippetparsed[snippet2id[v['snippet']]] = v['processed_snippet']

    def _add_retrieval_psgs_ids(qa_dataset, retrieval_dataset):
        def _helper(example, idx):
            example['retrieval_psgs_ids'], example['retrieval_psgs_scores'] = retrieval_dataset[idx]
            return example
        return qa_dataset.map(_helper, with_indices=True)
    
    dataset_train = _add_retrieval_psgs_ids(datasets['train'], retrieval_snippet_id_train)
    dataset_validation_seen = _add_retrieval_psgs_ids(datasets['validation_seen'], retrieval_snippet_id_dev_seen)
    dataset_validation_unseen = _add_retrieval_psgs_ids(datasets['validation_unseen'], retrieval_snippet_id_dev_unseen)
    dataset_test_seen = _add_retrieval_psgs_ids(datasets['test_seen'], retrieval_snippet_id_test_seen)
    dataset_test_unseen = _add_retrieval_psgs_ids(datasets['test_unseen'], retrieval_snippet_id_test_unseen)

    if data_args.debug_sharc:
        dataset_train = dataset_train.filter(lambda example, indice: indice < 32, with_indices=True)
        dataset_validation_seen = dataset_validation_seen.filter(lambda example, indice: indice < 16, with_indices=True)
        dataset_validation_unseen = dataset_validation_unseen.filter(lambda example, indice: indice < 16, with_indices=True)
        dataset_test_seen = dataset_test_seen.filter(lambda example, indice: indice < 16, with_indices=True)
        dataset_test_unseen = dataset_test_unseen.filter(lambda example, indice: indice < 16, with_indices=True)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = T5Tokenizer.from_pretrained(
        data_args.tokenizer_file if (data_args.tokenizer_file and len(os.listdir(data_args.tokenizer_file)) != 0) else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))

    model.config.max_length = model_args.max_target_length
    model.config.num_beams = model_args.num_beams
    
    def _preprocess_function_user_info(example):
        result = {
            'labels': [],
            'input_ids': [],
        }

        ssep_token_id = tokenizer.encode('<ssep>', add_special_tokens=False)[0]
        
        result['input_ids'] = tokenized_sharc_data['questions']['<qu> ' + example['question']] + [ssep_token_id]

        result['input_ids'].extend(tokenized_sharc_data['scenarios']['<sc> ' + example['scenario']] + [ssep_token_id])

        for _fqa in example['history']:
            result['input_ids'].extend(tokenized_sharc_data['follow_up_questions']['<fuq> ' + _fqa['follow_up_question']])
            if 'yes' in _fqa['follow_up_answer'].lower():
                result['input_ids'].extend(tokenized_sharc_data['follow_up_answers']["<fua> yes"]) 
            else:
                result['input_ids'].extend(tokenized_sharc_data['follow_up_answers']["<fua> no"]) 
            result['input_ids'].append(ssep_token_id)
            
        return result

    def _preprocess_function_snippet(result, top_k_snippet_ids, example, top_k_snippet_scores):
        
        ssep_token_id = tokenizer.encode('<ssep>', add_special_tokens=False)[0]
        cls_token_id = tokenizer.encode('<cls>', add_special_tokens=False)[0]
        sn_token_id = tokenizer.encode('<sn>', add_special_tokens=False)[0]

        max_top_k_ids = 0
        temp_input_ids = deepcopy(result['input_ids'])
        for snippet_id in top_k_snippet_ids:
            if snippet_id == snippet2id[example['snippet']]:
                m = tree_mapping_data[example['tree_id']]['processed_snippet']
            else:
                m = snippetid2snippetparsed[snippet_id]
            temp_input_ids += [sn_token_id]

            for clause_id, edus in enumerate(m['edu_t']):
                for edu_id, edu in enumerate(edus):
                    temp_input_ids += [cls_token_id] + edu
            temp_input_ids += [ssep_token_id]
            
            if len(temp_input_ids) <= config.n_positions - 1:
                max_top_k_ids += 1
            else:
                top_k_snippet_ids = top_k_snippet_ids[:max_top_k_ids]
                break

        top_k_snippet_scores = softmax(top_k_snippet_scores)

        for snippet_id, snippet_score in zip(top_k_snippet_ids, top_k_snippet_scores):
            if snippet_id == snippet2id[example['snippet']]:
                is_gold = True
                m = tree_mapping_data[example['tree_id']]['processed_snippet']
            else:
                is_gold = False
                m = snippetid2snippetparsed[snippet_id]
            result['input_ids'] += [sn_token_id]
            for clause_id, edus in enumerate(m['edu_t']):
                for edu_id, edu in enumerate(edus):
                    result['input_ids'] += [cls_token_id] + edu
            result['input_ids'] += [ssep_token_id]

        result['input_ids'] = result['input_ids'][:config.n_positions - 1]

        if len(result['input_ids']) == config.n_positions - 1:
            result['input_ids'][-1] = ssep_token_id
        result['input_ids'] += [tokenizer.eos_token_id]

        if example['label'].lower() == 'yes':
            result['labels'] = tokenizer("1", return_tensors="pt").input_ids.squeeze()
        elif example['label'].lower() == 'no':
            result['labels'] = tokenizer("2", return_tensors="pt").input_ids.squeeze()
        else:
            result['labels'] = tokenizer("3 " + example['label'], return_tensors="pt").input_ids.squeeze()

        result['attention_mask'] = [1 for _ in result['input_ids']]

        return result

    def preprocess_function_train(example):
        result = _preprocess_function_user_info(example)
        
        if snippet2id[example['snippet']] in example['retrieval_psgs_ids'][:data_args.top_k_snippets]:
            top_k_snippet_ids = example['retrieval_psgs_ids'][:data_args.top_k_snippets]
            top_k_snippet_scores = example['retrieval_psgs_scores'][:data_args.top_k_snippets]
        else:
            top_k_snippet_ids = [snippet2id[example['snippet']]] + example['retrieval_psgs_ids'][:data_args.top_k_snippets - 1]
            top_k_snippet_scores = [100.0] + example['retrieval_psgs_scores'][:data_args.top_k_snippets - 1]
            
        result = _preprocess_function_snippet(result, top_k_snippet_ids, example, top_k_snippet_scores)
        
        return result

    dataset_train = dataset_train.map(preprocess_function_train, load_from_cache_file=not data_args.overwrite_cache)
    dataset_train = dataset_train.remove_columns(["utterance_id", "tree_id", "source_url", "snippet", "question", "scenario", "answer", "history", "evidence", "retrieval_psgs_ids", "retrieval_psgs_scores", "label"])
    print(dataset_train)


    def preprocess_function_eval(example):
        result = _preprocess_function_user_info(example)
        top_k_snippet_ids = example['retrieval_psgs_ids'][:data_args.top_k_snippets]
        top_k_snippet_scores = example['retrieval_psgs_scores'][:data_args.top_k_snippets]
        result = _preprocess_function_snippet(result, top_k_snippet_ids, example, top_k_snippet_scores)
        return result

    dataset_validation_seen = dataset_validation_seen.map(preprocess_function_eval, load_from_cache_file=not data_args.overwrite_cache)
    dataset_validation_seen = dataset_validation_seen.remove_columns(["utterance_id", "tree_id", "source_url", "snippet", "question", "scenario", "answer", "history", "evidence", "retrieval_psgs_ids", "retrieval_psgs_scores", "label"])
    dataset_validation_unseen = dataset_validation_unseen.map(preprocess_function_eval, load_from_cache_file=not data_args.overwrite_cache)
    dataset_validation_unseen = dataset_validation_unseen.remove_columns(["utterance_id", "tree_id", "source_url", "snippet", "question", "scenario", "answer", "history", "evidence", "retrieval_psgs_ids", "retrieval_psgs_scores", "label"])
    dataset_test_seen = dataset_test_seen.map(preprocess_function_eval, load_from_cache_file=not data_args.overwrite_cache)
    dataset_test_seen = dataset_test_seen.remove_columns(["utterance_id", "tree_id", "source_url", "snippet", "question", "scenario", "answer", "history", "evidence", "retrieval_psgs_ids", "retrieval_psgs_scores", "label"])
    dataset_test_unseen = dataset_test_unseen.map(preprocess_function_eval, load_from_cache_file=not data_args.overwrite_cache)
    dataset_test_unseen = dataset_test_unseen.remove_columns(["utterance_id", "tree_id", "source_url", "snippet", "question", "scenario", "answer", "history", "evidence", "retrieval_psgs_ids", "retrieval_psgs_scores", "label"])
    dataset_validation = concatenate_datasets([dataset_validation_seen, dataset_validation_unseen])
    dataset_test = concatenate_datasets([dataset_test_seen, dataset_test_unseen])


    def compute_metrics(p: EvalPrediction):
        whole_preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        whole_labels = p.label_ids
        preds = []
        golds = []

        for idx in range(whole_preds.shape[0]):
            pred = {'utterance_id': idx}
            gold = {'utterance_id': idx}
            if whole_preds[idx, 1] == tokenizer.encode('1')[0]:
                pred['answer'] = 'yes'
            elif whole_preds[idx, 1] == tokenizer.encode('2')[0]:
                pred['answer'] = 'no'
            elif whole_preds[idx, 1] == tokenizer.encode('3')[0]:
                pred_temp = whole_preds[idx, 2:].copy()
                pred_temp[pred_temp == -100] = 0
                pred['answer'] = tokenizer.decode(pred_temp, skip_special_tokens=True)
            else:
                pred['answer'] = 'missing'

            if whole_labels[idx, 0] == tokenizer.encode('1')[0]:
                gold['answer'] = 'yes'
            elif whole_labels[idx, 0] == tokenizer.encode('2')[0]:
                gold['answer'] = 'no'
            else:
                gold_temp = whole_labels[idx, 1:].copy()
                gold_temp[gold_temp == -100] = 0
                gold['answer'] = tokenizer.decode(gold_temp, skip_special_tokens=True)

            preds.append(pred)
            golds.append(gold)

        import evaluator
        with NamedTemporaryFile('w') as fp, NamedTemporaryFile('w') as fg:
            json.dump(preds, fp)
            fp.flush()
            json.dump(golds, fg)
            fg.flush()
            results = evaluator.evaluate(fg.name, fp.name, mode='combined')
        return results

    datacollator = DataCollatorForUnify(
        tokenizer = tokenizer,
        model = model,
        padding = True,
        max_length = config.n_positions,
        return_tensors = 'pt'
    )

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, correct_bias=True)
    lr_scheduler = get_constant_schedule(optimizer, last_epoch=-1)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_validation_seen if data_args.debug_sharc else dataset_train,
        eval_dataset=dataset_validation_seen if data_args.debug_sharc else dataset_validation,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        tokenizer=tokenizer,
        data_collator=datacollator,
        optimizers = (optimizer, lr_scheduler)
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    if training_args.do_eval:
        if training_args.do_train:
            logger.info(
                f"Loading best model from {trainer.state.best_model_checkpoint} (score: {trainer.state.best_metric})."
            )

            trainer.model = trainer.model.from_pretrained(
                trainer.state.best_model_checkpoint,
            )
            if not trainer.is_model_parallel:
                trainer.model = trainer.model.to(trainer.args.device)

            tasks = ['dev', 'test']
            eval_datasets = [dataset_validation, dataset_test]
        else:
            tasks = ['dev', 'test', 'dev_seen', 'dev_unseen', 'test_seen', 'test_unseen']
            eval_datasets = [dataset_validation, dataset_test, dataset_validation_seen, dataset_validation_unseen, dataset_test_seen, dataset_test_unseen]

        for eval_dataset, task in zip(eval_datasets, tasks):
            logger.info(f"*** {task} ***")
            output = trainer.predict(test_dataset=eval_dataset)
            eval_result = output.metrics
            eval_prediction = output.predictions
            eval_result['best_ckpt'] = trainer.state.best_model_checkpoint
            eval_result['best_ckpt_metric'] = trainer.state.best_metric
            result_eval_file = os.path.join(training_args.output_dir, f"results_{task}.json")
            prediction_eval_file = os.path.join(training_args.output_dir, f"predictions_{task}.npy")
            if trainer.is_world_process_zero():
                logger.info(f"***** Eval results {task} *****")
                for key, value in sorted(eval_result.items()):
                    logger.info(f"  {key} = {value}")
                with open(result_eval_file, "w") as f:
                    json.dump(eval_result, f)
                np.save(prediction_eval_file, eval_prediction)


if __name__ == "__main__":
    main()
