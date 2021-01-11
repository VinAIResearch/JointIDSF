lr_list=(1e-6 2e-6 4e-6 1e-5 2e-5 4e-5)
for lr in "${lr_list[@]}" 
do
echo "${lr}"
export MODEL_DIR=atis_bert_crf
export MODEL_DIR=$MODEL_DIR"/"$lr
echo "${MODEL_DIR}"
/usr/bin/python3.7 main.py --task atis \
                  --model_type bert \
                  --model_dir $MODEL_DIR \
                  --data_dir data \
                  --do_train \
                  --do_eval \
                  --num_train_epochs 100 \
                  --use_crf \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --tuning_metric mean_intent_slot \
                  --learning_rate $lr
done