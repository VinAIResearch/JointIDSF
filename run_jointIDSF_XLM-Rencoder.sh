#As we initialize JointIDSF from JointBERT, user need to train a base model JointBERT first
./run_jointBERT-CRF_XLM-Rencoder.sh
#Train JointIDSF
export lr=3e-5
export c=0.25
export s=10
echo "${lr}"
export MODEL_DIR=JointIDSF_XLM-Rencoder
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c"/"$s
echo "${MODEL_DIR}"
python3 main.py --token_level syllable-level \
                  --model_type xlmr \
                  --model_dir $MODEL_DIR \
                  --data_dir PhoATIS \
                  --seed $s \
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
                  --pretrained \
                  --pretrained_path JointBERT-CRF_XLM-Rencoder/4e-5/0.45/10 \
                  --learning_rate $lr