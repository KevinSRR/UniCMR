#!/usr/bin/env python
import os
import torch
import string
import json
from tqdm import tqdm
import editdistance

from transformers import T5Tokenizer
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

import numpy as np
from datasets import load_dataset, concatenate_datasets

from sklearn.metrics import accuracy_score, confusion_matrix

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process

MATCH_IGNORE = {'do', 'did', 'does',
                'is', 'are', 'was', 'were', 'have', 'will', 'would',
                '?', }
PUNCT_WORDS = set(string.punctuation)


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
        default=None, metadata={"help": "A csv or a json file containing the validation seen tfidf ata."}
    )
    validation_retrieval_unseen_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation unseen tfidf data."}
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
        default=False,
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


def _decode(tokenizer, doc):
    decoded = tokenizer.decode(doc, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip('\n').strip()
    return decoded

def filter_token(tokenizer, text):
    filtered_text = []
    for token_id in text:
        if _decode(tokenizer, token_id).lower() not in MATCH_IGNORE and _decode(tokenizer, token_id).strip() != "":
            filtered_text.append(token_id)
    return _decode(tokenizer, filtered_text)

def merge_edus(id2snippet_parsed):
    for id, snippet_parsed in id2snippet_parsed.items():
        if snippet_parsed['has_edu']:
            id2snippet_parsed[id]['edus'] = [_merge_edus(_edu) for _edu in snippet_parsed['edus']] 
    return id2snippet_parsed

def _merge_edus(edus):
    special_toks = ['if ', 'and ', 'or ', 'to ', 'unless ', 'but ', 'as ', 'except ']
    special_puncts = ['.', ':', ',',]
    spt_idx = []
    for idx, edu in enumerate(edus):
        if idx == 0:
            continue
        is_endwith = False
        for special_punct in special_puncts:
            if edus[idx-1].strip().endswith(special_punct):
                is_endwith = True
        is_startwith = False
        for special_tok in special_toks:
            if edu.startswith(special_tok):
                is_startwith = True
        if (not is_endwith) and (not is_startwith):
            spt_idx.append(idx)
    edus_spt = []
    for idx, edu in enumerate(edus):
        if idx not in spt_idx or idx == 0:
            edus_spt.append(edu)
        else:
            edus_spt[-1] += ' ' + edu
    return edus_spt


def _extract_edus(snippet_parsed, title_tokenized, sentences_tokenized, tokenizer):
    
    edus_tokenized = []

    if snippet_parsed['title'].strip('\n').strip() != '':
        edus_tokenized.append([title_tokenized])

    if snippet_parsed['is_bullet']:
        for sentence_tokenized in sentences_tokenized:
            edus_tokenized.append([sentence_tokenized])
    else:
        for idx_sentence in range(len(sentences_tokenized)):
            edus_tokenized_i = []
           
            current_edus = snippet_parsed['edus'][idx_sentence] 
            current_sentence_tokenized = sentences_tokenized[idx_sentence] 

            p_start, p_end = 0, 0
            for edu in current_edus:
                edu = edu.strip('\n').strip().replace(' ', '').lower()
                
                edu = edu.replace('§', '')
                if ('``' in edu) and ('\'\'' in edu):
                    edu = edu.replace('``', '"').replace('\'\'', '"')
                
                for p_sent in range(p_start, len(current_sentence_tokenized)):
                    sent_span = _decode(tokenizer, current_sentence_tokenized[p_start:p_sent+1]).replace(' ', '').lower()
                    if edu == sent_span:
                        p_end = p_sent
                        edus_tokenized_i.append(current_sentence_tokenized[p_start:p_end + 1])
                        p_start = p_end + 1
                        break
            
    
            assert p_end == len(current_sentence_tokenized) - 1
            edus_tokenized.append(edus_tokenized_i)

    return edus_tokenized


def extract_edus(fuqs, snippet_parsed, tokenized_data, tokenizer):

    output = {}
    
    if snippet_parsed['title'].strip('\n').strip() != '':
        title_tokenized = tokenized_data['titles'][snippet_parsed['title']]
    else:
        title_tokenized = None
    sentences_tokenized = [tokenized_data['clauses'][s] for s in snippet_parsed['clauses']]
    output['clause_t'] = [title_tokenized] + sentences_tokenized if title_tokenized else sentences_tokenized
    output['edu_t'] = _extract_edus(snippet_parsed, title_tokenized, sentences_tokenized, tokenizer)

    return output

def _get_tokenized_kv(original, tokenized):
    out = {}
    for _q, _v in zip(original, tokenized['input_ids']):
        out[_q] = _v
    return out

def _load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def _loads_json(path):
    with open(path, 'r', ) as f:
        dataset = []
        for idx, line in enumerate(f):
            example = json.loads(line)
            dataset.append(example)
    return dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    id2snippet = _load_json(data_args.snippet_file)
    
    id2snippet_parsed = _load_json(data_args.snippet_file.replace(".json", "_parsed.json"))

    snippet2id = {v: k for k, v in id2snippet.items()}
    
    dataset_train = _loads_json(data_args.train_qa_file)
    
    dataset_validation_seen = _loads_json(data_args.validation_qa_seen_file)
    
    dataset_validation_unseen = _loads_json(data_args.validation_qa_unseen_file)
    
    dataset_test_seen = _loads_json(data_args.test_qa_seen_file)
    
    dataset_test_unseen = _loads_json(data_args.test_qa_unseen_file)
    
    dataset_all = sum([dataset_train, dataset_validation_seen, dataset_validation_unseen, dataset_test_seen, dataset_test_unseen], []) 

    id2snippet_parsed = merge_edus(id2snippet_parsed)

    tokenizer = T5Tokenizer.from_pretrained(
        data_args.tokenizer_file if (data_args.tokenizer_file and len(os.listdir(data_args.tokenizer_file)) !=0) else model_args.model_name_or_path, 
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision, 
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if not (data_args.tokenizer_file and len(os.listdir(data_args.tokenizer_file)) !=0):
        tokenizer.add_tokens(['<qu>', '<sc>', '<sn>', '<fuq>', '<fua>', '<fqa>', '<ssep>', '<cls>'], special_tokens=True)

    tokenized_sharc_path = f'./data/{model_args.model_name_or_path}-tokenized.json'
    
    if os.path.exists(tokenized_sharc_path):
        with open(tokenized_sharc_path) as f:
            tokenized_sharc_data = json.load(f)
    else:
        os.makedirs('./data', exist_ok=True)
        questions = list(set(["<qu> " + _ex['question'] for _ex in dataset_all]))
        scenarios = list(set(["<sc> " + _ex['scenario'] for _ex in dataset_all]))
        follow_up_questions = list(set(
            ["<fuq> " + fuqa['follow_up_question'] for _ex in dataset_all for fuqa in _ex['history']] +
            ["<fuq> " + fuqa['follow_up_question'] for _ex in dataset_all for fuqa in _ex['evidence']]))
        follow_up_answers = ["<fua> yes", "<fua> no", ]
        inquire_answers = list(set([_ex['answer'] for _ex in dataset_all]))
        clauses, titles = [], []
        for _ex in id2snippet_parsed.values():
            clauses.extend(_ex['clauses'])
            titles.append(_ex['title'])
        clauses = list(set(clauses))
        titles = list(set(titles))

        questions_tokenized = tokenizer(questions, add_special_tokens=False)
        scenarios_tokenized = tokenizer(scenarios, add_special_tokens=False)
        follow_up_questions_tokenized = tokenizer(follow_up_questions, add_special_tokens=False)
        follow_up_answers_tokenized = tokenizer(follow_up_answers, add_special_tokens=False)
        inquire_answers_tokenized = tokenizer(inquire_answers, add_special_tokens=False)
        clauses_tokenized = tokenizer(clauses, add_special_tokens = False)
        titles_tokenized = tokenizer(titles, add_special_tokens = False)
        
        
        tokenized_sharc_data = {
            'questions': _get_tokenized_kv(questions, questions_tokenized),
            'scenarios': _get_tokenized_kv(scenarios, scenarios_tokenized),
            'follow_up_questions': _get_tokenized_kv(follow_up_questions, follow_up_questions_tokenized),
            'follow_up_answers': _get_tokenized_kv(follow_up_answers, follow_up_answers_tokenized),
            'inquire_answers': _get_tokenized_kv(inquire_answers, inquire_answers_tokenized),
            'clauses': _get_tokenized_kv(clauses, clauses_tokenized),
            'titles': _get_tokenized_kv(titles, titles_tokenized),
        }
        
        with open(tokenized_sharc_path, 'w') as f:
            json.dump(tokenized_sharc_data, f)
        print(f"Saving tokenized sharc data {tokenized_sharc_path}")

    tree2fuq = {}

    tree2snippet = {}

    for _ex in dataset_all:
        if _ex['tree_id'] not in tree2fuq:
            tree2fuq[_ex['tree_id']] = set()
        for h in _ex['history'] + _ex['evidence']:
            tree2fuq[_ex['tree_id']].add(h['follow_up_question'])
        if _ex['answer'].lower() not in ['yes', 'no', 'irrelevant']:
            tree2fuq[_ex['tree_id']].add(_ex['answer'])
        if 'tree_id' not in tree2snippet:
            tree2snippet[_ex['tree_id']] = _ex['snippet']
        else:
            assert tree2snippet[_ex['tree_id']] == _ex['snippet'], f"{tree2snippet[_ex['tree_id']]}\n{_ex['snippet']}"

    processed_tree_path = f'./data/{model_args.model_name_or_path}-tree-mapping.json'


    processed_tree = {
        k: {
            "processed_snippet": extract_edus(tree2fuq[k], id2snippet_parsed[snippet2id[tree2snippet[k]]], tokenized_sharc_data, tokenizer),
            "follow_up_questions": list(tree2fuq[k]),
            "snippet": tree2snippet[k],
        } for k in tqdm(tree2fuq.keys())
    }

    print(f'saving {processed_tree_path}')
    with open(processed_tree_path, 'w') as f:
        json.dump(processed_tree, f)

    if not (data_args.tokenizer_file and len(os.listdir(data_args.tokenizer_file)) != 0): 
        tokenizer.save_pretrained(data_args.tokenizer_file)
        
if __name__ == "__main__":
    main()