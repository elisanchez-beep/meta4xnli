#!/bin/bash
#SBATCH --job-name=xlm-rob
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/nli_train_rob.log
#SBATCH --error=.slurm/nli_train_rob.err

#export CUDA_VISIBLE_DEVICES=3

#source /ikerlariak/ragerri/transformers-4.20/bin/activate.csh
#source /ikerlariak/ragerri/tximista-transformers-4.20/bin/activate.csh

rm -r -f ~/.cache/huggingface/datasets/
rm -r -f /gaueko1/HF_datasets_cache/xnli/en

SEED=70
NUM_EPOCHS=4
BATCH_SIZE=8
GRADIENT_ACC_STEPS=2
BATCH_SIZE_PER_GPU=$(( $BATCH_SIZE*$GRADIENT_ACC_STEPS ))
LEARN_RATE=0.00001 #4/5
WARMUP=0.06
WEIGHT_DECAY=0.1
MAX_SEQ_LENGTH=512

MODEL='xlm-roberta-large'
OUTPUT_DIR='./xlm-roberta-large/en'
LOGGING_DIR='./logs/xlm-roberta.log'
DIR_NAME='xnli_no_met'_${BATCH_SIZE_PER_GPU}_${WEIGHT_DECAY}_${LEARN_RATE}_$(date +'%m-%d-%y_%H-%M')

python3 ./bsc_run_glue.py --model_name_or_path $MODEL --seed $SEED\
                                        --dataset_script_path ./meta4xnli_int.py --dataset_config_name en \
                                        --train_file './xnli.test.tsv' \
                                        --validation_file './xnli.dev.tsv' \
                                        --test_file './xnli.test.tsv' \
                                        --task_name mnli --do_train --do_eval --do_predict \
                                        --num_train_epochs $NUM_EPOCHS --gradient_accumulation_steps $GRADIENT_ACC_STEPS --per_device_train_batch_size $BATCH_SIZE \
                                        --learning_rate $LEARN_RATE --max_seq_length $MAX_SEQ_LENGTH \
                                        --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY \
                                        --output_dir $OUTPUT_DIR/$DIR_NAME --overwrite_output_dir \
                                        --logging_dir $LOGGING_DIR/$DIR_NAME --logging_strategy epoch \
                                        --overwrite_cache \
                                        --metric_for_best_model accuracy --save_strategy epoch --evaluation_strategy epoch --load_best_model_at_end

