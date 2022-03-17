export lr=3e-5
export c_intent=0.33
export c_slot=0.33
export s=42

echo "${lr}"
export MODEL_DIR=JointDF-CRF_PhoBERTencoder
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c_intent"/"$s
echo "${MODEL_DIR}"
python3 main.py --token_level syllable \
                  --model_type jointdf \
                  --model_dir $MODEL_DIR \
                  --data_dir PhoDF \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 10 \
                  --tuning_metric mean_intent_slot \
                  --use_crf \
                  --gpu_id 0 \
                  --embedding_type soft \
                  --intent_loss_coef $c_intent \
		  --slot_loss_coef $c_slot \
                  --learning_rate $lr