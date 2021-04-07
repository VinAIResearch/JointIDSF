export lr=4e-5
export c=0.15
export s=1
echo "${lr}"
export MODEL_DIR=viatis_phobert_crf
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c"/"$s
echo "${MODEL_DIR}"
/usr/bin/python3.7 main.py --token_level word-level \
                  --model_type phobert \
                  --model_dir $MODEL_DIR \
                  --data_dir data \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 50 \
                  --tuning_metric mean_intent_slot \
                  --use_crf \
                  --gpu_id 0 \
                  --embedding_type soft \
                  --intent_loss_coef $c \
                  --learning_rate $lr