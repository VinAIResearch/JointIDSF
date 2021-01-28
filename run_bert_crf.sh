lr_list=(1e-5 2e-5 3e-5 4e-5 5e-5)
for lr in "${lr_list[@]}" 
do
echo "${lr}"
export MODEL_DIR=atisthinh_bert_crf_attn_2
export MODEL_DIR=$MODEL_DIR"/"$lr
echo "${MODEL_DIR}"
/usr/bin/python3.7 main.py --task atis-thinh \
                  --model_type bert \
                  --model_dir $MODEL_DIR \
                  --data_dir data \
                  --do_train \
                  --do_eval \
                  --num_train_epochs 1000 \
                  --use_crf \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --tuning_metric mean_intent_slot \
                  --use_intent_context_attention \
                  --attention_embedding_size 768 \
                  --slot_loss_coef 2 \
                  --learning_rate $lr
done
