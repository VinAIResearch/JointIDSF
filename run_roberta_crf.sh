lr_list=(1e-5 2e-5 3e-5 4e-5 5e-5)
for lr in "${lr_list[@]}" 
do
echo "${lr}"
export MODEL_DIR=atis_roberta_crf_attn_256
export MODEL_DIR=$MODEL_DIR"/"$lr
echo "${MODEL_DIR}"
/usr/bin/python3.7 main.py --task atis \
                  --model_type roberta \
                  --model_dir $MODEL_DIR \
                  --data_dir data \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 1000 \
                  --tuning_metric mean_intent_slot \
                  --use_intent_context_attention \
                  --intent_embedding_size 256 \
                  --attention_embedding_size 256 \
                  --use_crf \
                  --learning_rate $lr
done
