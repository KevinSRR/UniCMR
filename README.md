# UniCMR

This is the source code for the paper [Towards End-to-End Open Conversational Machine Reading](https://arxiv.org/abs/2210.07113). 

(Codes are being further cleaned and updated)

## 1. Datasets

Please refer to [MUDERN](https://github.com/Yifan-Gao/open_retrieval_conversational_machine_reading) and [OSCAR](https://github.com/ozyyshr/OSCAR) for preparing the OR-CMR raw datasets under the folder  `./data` and then begin the following processing steps.

## 2. Discourse Segmentation

For convenience, we make a discourse segmented version of our rule text knowledge base beforehand.

### Requirement

- Pytorch==0.4.1
- NLTK==3.4.5
- numpy==1.18.1
- pycparser==2.20
- six==1.14.0
- tqdm==4.44.1

### Instruction

1. Run `cd segedu`
2. Run `pip install -r requirements.txt`
3. Run `python open_sharc_discourse_segmentation.py`

## 3. TF-IDF Retrieval

For convenience, we make a retrieved rule texts for every single rule text beforehand.

### Requirement

- numpy
- scikit-learn
- regex
- tqdm
- Scipy
- NLTK
- elasticsearch
- pexpect==4.2.1

### Instruction

1. Run `pip install -r requirements.txt`

2. Build Sqlite DB via: 

   Here base_dir=./data

   db_path =`./data/sharc_raw/json/sharc_open_id2snippet.json`

   ```
   mkdir -p {base_dir}/tfidf
   python3 build_db.py ${db_path} ${base_dir}/tfidf/db.db --num-workers 60`.
   ```

3. Run the following command to build TF-IDF index:

   ```
   python3 build_tfidf.py ${base_dir}/tfidf/db.db ${base_dir}/tfidf
   ```

   It will save TF-IDF index in `${base_dir}/tfidf`

4. Run inference code to save retrieval results.

   ```
   bash inference_tfidf.sh
   ```

## 4. Preprocess

Tokenize the user information and construct the dialogue tree.

### Requirement

- Python 3.6
- Pytorch (1.6.0)
- NLTK (3.4.5)
- spacy (2.0.16)
- transformers (4.3.2)

### Instruction

1. Run `cd ./UniCMR`
2. Run `pip install -r requirements.txt`
3. Run `bash preprocess.sh`

## 5. Decision Making and Question Generation

Training and inference of our UniCMR.

### Requirement

- Python 3.6
- Pytorch (1.6.0)
- NLTK (3.4.5)
- spacy (2.0.16)
- transformers (4.3.2)

### Instruction

1. Run `cd ./UniCMR`
2. Run `pip install -r requirements.txt`
3. Run `bash run.sh`



## Acknowledgement

Part of our codes are borrowed from the [codes](https://github.com/Yifan-Gao/open_retrieval_conversational_machine_reading) of [Open-Retrieval Conversational Machine Reading](https://arxiv.org/abs/2102.08633), many thanks.