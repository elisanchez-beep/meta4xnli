# Meta4XNLI

This repository contains the data and code from our paper _Meta4XNLI: A Crosslingual Parallel Corpus for Metaphor Detection and Interpretation_. Meta4XNLI is a parallel dataset with annotations in English and Spanish for metaphor detection at token level and metaphor interpretation framed within NLI the task. We evaluated multilingual models such as mDeBERTa and XLM-RoBERTa-large to examine their performance in monolingual, cross-lingual, cross-domain and zero-shot scenarios for metaphor detection. Also, we evaluated if these models struggle to identify inference relationships when metaphors are involved in the text.


## Content
The repository is organised as follows:

### `data/`

- `cometa/`: train and test partitions in tabulated format.
- `vua/`: train, dev and test partitions of VUA-20 version with all POS labeled [shared task 2020] in tabulated format.
- `meta4xnli/`
	- `detection/`: all files in tabulated format.
		  -`source_datasets/`: files in tabulated format split by the original dataset, premises and hypotheses separated, in English `en/` and Spanish `es/`.
		  - `splits/`: train, dev and test splits to train and evaluate in English `en/` and Spanish `es/`.	
	- `interpretation/`: all files in .tsv format with following fields: `{language}`: en or es; `{gold_label}`: inference label from original dataset: [entailment, neutral or contradiction]; `{sentence1}`: premise; `{sentence2}`: hypothesis; `{promptID}`: premise identifier number_{source_dataset}; `{pairID}`: premise and hypothesis          pair identifier number_{source_dataset}; `{genre}`: text domain labeled from original dataset annotations; `{source_dataset}`: original dataset to which the pair belongs.
	    - `source_datasets/`: each file includes sentences in English and Spanish: `{source_dataset}_met.tsv`: files with pairs with metaphors; `{source_dataset}_no_met.tsv`: files with pairs without metaphors.
	    - `splits`: train, dev and test splits with and without metaphors, both languages in the same file. The language must be specified in argument `dataset_config_name` of `fine-tuning-nli.sh` file.



### `scripts/`
- `generate_scripts.py`: create scripts with different combinations of parameters.
- `meta4xnli_det/int.py`: script to process dataset files. You need to set your paths in variables `_TRAIN_DATA_URL, _EVAL_DATA_URL, _TESTEVAL_DATA_URL` (in meta4xnli_int.py); `_TRAINING_FILE, _DEV_FILE, _TEST_FILE` (in meta4xnli_det.py).   
- `interpretation/bsc_run_glue.py`: script to fine-tune or evaluate models for the task of NLI.
- `detection/bsc_run_ner.py`: script to fine-tune or evaluate models for the task of sequence labeling.
- `fine-tune-det/nli.sh`: files to execute fine-tuning scripts with corresponding parameters.
- `inference-nli.sh/inference-metaphor-detection.sh`: files to evaluate trained models with corresponding parameters.




## Run experiments:
1. Install Transformers 4.20
2. Place the chosen train, dev and test files from data/ in the path of the `meta4xnli_det/int.py` scripts.
3. Generate the scripts using generate_scripts.py from `detection/` or `interpretation/` folders depending on the desired task
4. Run (you can just run the best models using the hyperparameters specified in
the paper.

#### Contact: {rodrigo.agerri, elisa.sanchez}@ehu.eus
