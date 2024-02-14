#!/bin/bash
#SBATCH --job-name=deberta_met_xnli_train
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/deberta_met_xnli_train.log
#SBATCH --error=.slurm/deberta_met_xnli_train.err

SEED=75
NUM_EPOCHS=4
BATCH_SIZE=8
GRADIENT_ACC_STEPS=1
BATCH_SIZE_PER_GPU=$(( $BATCH_SIZE*$GRADIENT_ACC_STEPS ))
LEARN_RATE=0.00005
WARMUP=0.06
WEIGHT_DECAY=0.01


MODEL='microsoft/mdeberta-v3-base'
OUTPUT_DIR='./outputs/en_es_multilingual/mdeberta-v3-base-output'
LOGGING_DIR='./logs/mdeberta-v3-base.log'
DIR_NAME='met_xnli_test'_${BATCH_SIZE_PER_GPU}_${WEIGHT_DECAY}_${LEARN_RATE}_$(date +'%m-%d-%y_%H-%M')

python3 ,/bsc_run_ner.py --model_name_or_path $MODEL --seed $SEED \
                                         --dataset_script_path ./meta4xnli_det.py \
                                         --task_name ner --do_train --do_eval --do_predict \
                                         --num_train_epochs $NUM_EPOCHS --gradient_accumulation_steps $GRADIENT_ACC_STEPS --per_device_train_batch_size $BATCH_SIZE \
                                         --learning_rate $LEARN_RATE \
                                         --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY \
                                         --output_dir $OUTPUT_DIR/$DIR_NAME --overwrite_output_dir \
                                         --logging_dir $LOGGING_DIR/$DIR_NAME --logging_strategy epoch \
                                         --overwrite_cache \
                                         --metric_for_best_model f1 --save_strategy epoch --evaluation_strategy epoch --load_best_model_at_end
