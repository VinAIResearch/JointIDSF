seed_list=(1 2 3 4 5)
for s in "${seed_list[@]}" 
do
echo "${s}"
export MODEL_DIR=viatis_mbert_crf_attn_200
export lr=2e-5
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$s
echo "${MODEL_DIR}"
/usr/bin/python3.7 main.py --task vi-atis-fix \
                  --model_type mbert \
                  --model_dir $MODEL_DIR \
                  --data_dir data \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 1000 \
                  --tuning_metric mean_intent_slot \
                  --use_intent_context_attention \
                  --attention_embedding_size 200 \
                  --use_crf \
                  --gpu_id 1 \
                  --intent_loss_coef 0.3 \
                  --learning_rate $lr
done