# rm data/cache*
seed_list=(1 2 3 4 5)
lr_list=(5e-5)
coef_list=(0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9)
# for s in "${seed_list[@]}" 
for lr in "${lr_list[@]}"
do
for c in "${coef_list[@]}"
do
echo "${lr}"
export MODEL_DIR=viatis_xlmr_crf_attn_200
# export lr=2e-5
# export s=1
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c
echo "${MODEL_DIR}"
/usr/bin/python3.7 main.py --task vi-atis-fix \
                  --model_type xlmr \
                  --model_dir $MODEL_DIR \
                  --data_dir data \
                  --seed 1 \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 50 \
                  --tuning_metric mean_intent_slot \
                  --use_intent_context_attention \
                  --attention_embedding_size 200 \
                  --use_crf \
                  --gpu_id 0 \
                  --embedding_type soft \
                  --intent_loss_coef $c \
                  --early_stopping 30 \
                  --learning_rate $lr
done
done