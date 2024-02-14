#!/bin/bash
#SBATCH --job-name=inference_met_det
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/inference_met_det.log
#SBATCH --error=.slurm/inference_met_det.err

#export CUDA_VISIBLE_DEVICES=2

MODEL=""
OUTPUT_DIR=""
DATASET_SCRIPT_PATH="./meta4xnli_det.py"


python ./bsc_run_ner.py --model_name_or_path $MODEL \
                                         --dataset_script_path $DATASET_SCRIPT_PATH \
                                         --task_name ner --do_eval --do_predict \
                                         --output_dir $OUTPUT_DIR/ --overwrite_output_dir \

