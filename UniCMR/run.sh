#!/usr/bin/env bash

OPENS=... # PATH for you to accomodate your Python
PYT=${OPENS}/bin/python


function train() {
  echo -----------------script arguments: SEED=$1 MAX_TARGET_LENGTH=$2 LR=$3 TOPK=$4
  export OMP_NUM_THREADS=1
  MODEL=t5-large
  TOPK=$4
  BZ=4
  MASPOT=22
  EPOCH=16
  LR=$3
  SEED=$1
  MAX_TARGET_LENGTH=$2
  NUM_BEAMS=5
  grad_accumulation_steps=8
  OUT="uniCMR_${MODEL}_BZ${BZ}_EP${EPOCH}_LR${LR}_TOP${TOPK}_seed${SEED}_TARLEN${MAX_TARGET_LENGTH}_GradAcc${grad_accumulation_steps}_NB${NUM_BEAMS}"

  echo '=============================='
  echo "=====$OUT====="
  echo '=============================='

  CUDA_VISIBLE_DEVICES=1 $PYT main.py \
  --tokenizer_file=./tokenizer/$MODEL \
  --train_qa_file=../data/sharc_raw/json/sharc_open_train.jsonl \
  --validation_qa_seen_file=../data/sharc_raw/json/sharc_open_dev_seen.jsonl \
  --validation_qa_unseen_file=../data/sharc_raw/json/sharc_open_dev_unseen.jsonl \
  --test_qa_seen_file=../data/sharc_raw/json/sharc_open_test_seen.jsonl \
  --test_qa_unseen_file=../data/sharc_raw/json/sharc_open_test_unseen.jsonl \
  --train_retrieval_file=../data/tfidf/train.json \
  --validation_retrieval_seen_file=../data/tfidf/dev_seen.json \
  --validation_retrieval_unseen_file=../data/tfidf/dev_unseen.json \
  --test_retrieval_seen_file=../data/tfidf/test_seen.json \
  --test_retrieval_unseen_file=../data/tfidf/test_unseen.json \
  --snippet_file=../data/sharc_raw/json/sharc_open_id2snippet.json \
  --tokenized_file=./data/${MODEL}-tokenized.json \
  --tree_mapping_file=./data/${MODEL}-tree-mapping.json \
  --top_k_snippets=$TOPK \
  --model_name_or_path=$MODEL \
  --output_dir=./out/${OUT} \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --evaluation_strategy=epoch \
  --per_device_train_batch_size=$BZ \
  --per_device_eval_batch_size=$BZ \
  --learning_rate=${LR} \
  --weight_decay=0.01 \
  --max_grad_norm=1.0 \
  --gradient_accumulation_steps=${grad_accumulation_steps} \
  --num_train_epochs=${EPOCH} \
  --warmup_steps=16 \
  --logging_steps=16 \
  --seed=${SEED} \
  --fp16 \
  --metric_for_best_model=fscore_bleu_4 \
  --predict_with_generate \
  --save_total_limit=1 \
  --remove_unused_columns=True \
  --ignore_pad_token_for_loss \
  --load_best_model_at_end \
  --overwrite_cache \
  --max_target_length=${MAX_TARGET_LENGTH} \
  --num_beams=${NUM_BEAMS}
}

seed=$1
max_tar_len=$2
lr=$3
topk=$4

train $seed $max_tar_len $lr $topk