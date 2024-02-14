#!/bin/bash
#SBATCH --job-name=inference_met_det
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/inference.log
#SBATCH --error=.slurm/inference.err

#export CUDA_VISIBLE_DEVICES=1

#source /ikerlariak/ragerri/transformers-4.20/bin/activate.csh

LANG='en'

DIR_NAME='xnli_dev_met'

MODEL=''
OUTPUT_DIR='/'$LANG'/xlm-roberta'

python3 ./bsc_run_glue.py --model_name_or_path $MODEL \
                                         --task_name mnli \
                                         --dataset_script_path ./xnli.py --dataset_config_name $LANG \
                                         --train_file './xnli.train.tsv' \
                                         --validation_file './xnli.dev.tsv' \
                                         --test_file './xnli.test.tsv' \
                                         --do_eval --do_predict \
                                         --output_dir $OUTPUT_DIR/$DIR_NAME --overwrite_output_dir \
                                         --overwrite_cache \
