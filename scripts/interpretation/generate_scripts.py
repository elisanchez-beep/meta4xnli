
bl = "\\"
dir_name = "DIR_NAME='xnli'_${BATCH_SIZE_PER_GPU}_${WEIGHT_DECAY}_${LEARN_RATE}_$(date +'%m-%d-%y_%H-%M')"
models = [{'m':'mdeberta-v3-base'}, {'m':'xlm-roberta-large', 'grad':2} ]
names = {'mdeberta-v3-base':'microsoft/mdeberta-v3-base','xlm-roberta-large':'xlm-roberta-large'}

batches = [8, 16, 32]
weight_dec = ['0.1', '0.01']
learn_rate = ['0.00001','0.00002','0.00003', '0.00005']

for model_d in models:
    model = model_d['m']
    name = names[model]
    grad = model_d['grad'] if 'grad' in model_d else 1
    for batch in batches:
        batch_noGrad = int(batch / grad)
        for weight in weight_dec:
            for lr in learn_rate:
                with open(f"{model}_xnli_batch{batch}_lr{lr}_decay{weight}.sh", "w") as f:
                    f.write(f"""#!/bin/bash
SEED=42
NUM_EPOCHS=4
BATCH_SIZE={batch_noGrad}
GRADIENT_ACC_STEPS={grad}
BATCH_SIZE_PER_GPU=$(( $BATCH_SIZE*$GRADIENT_ACC_STEPS ))
LEARN_RATE={lr}
WARMUP=0.06
WEIGHT_DECAY={weight}
MAX_SEQ_LENGTH=512

MODEL='{name}'
OUTPUT_DIR='./outputs/{model}-output'
LOGGING_DIR='./logs/{model}.log'
{dir_name}

python ../bsc_run_glue.py --model_name_or_path $MODEL --seed $SEED {bl}
                                         --dataset_script_path ./meta4xnli_int.py --dataset_config_name es {bl}
                                         --task_name mnli --do_train --do_eval --do_predict {bl}
                                         --num_train_epochs $NUM_EPOCHS --gradient_accumulation_steps $GRADIENT_ACC_STEPS --per_device_train_batch_size $BATCH_SIZE {bl}
                                         --learning_rate $LEARN_RATE --max_seq_length $MAX_SEQ_LENGTH {bl}
                                         --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY {bl}
                                         --output_dir $OUTPUT_DIR/$DIR_NAME --overwrite_output_dir {bl}
                                         --logging_dir $LOGGING_DIR/$DIR_NAME --logging_strategy epoch {bl}
                                         --overwrite_cache {bl}
                                         --metric_for_best_model accuracy --save_strategy epoch --evaluation_strategy epoch --load_best_model_at_end
                """)
