# lr_list=(1e-5 2e-5)
# coef_list=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)
# lr_list=(2e-5 3e-5 4e-5 5e-5)
# coef_list=(0.1 0.15)
#for s in "${seed_list[@]}"
# for lr in "${lr_list[@]}"
# do
# for c in "${coef_list[@]}"
# do
export lr=4e-5
export c=0.15
export s=100
echo "${lr}"
export MODEL_DIR=viatis_phobert_crf_attn
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c"/"$s
echo "${MODEL_DIR}"
/usr/bin/python3.7 main.py --task vi-atis-fix \
                  --model_type phobert \
                  --model_dir $MODEL_DIR \
                  --data_dir data \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --do_eval_dev \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 50 \
                  --tuning_metric mean_intent_slot \
                  --use_intent_context_attention \
                  --attention_embedding_size 200 \
                  --use_crf \
                  --token_level word \
                  --gpu_id 0 \
                  --embedding_type soft \
                  --intent_loss_coef $c \
                  --early_stopping 50 \
                  --pretrained \
                  --pretrained_path viatis_phobert_crf_sm/3e-5/0.6/100 \
                  --learning_rate $lr