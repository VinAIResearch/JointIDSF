lr_list = (1e-6, 2e-6, 4e-6, 1e-5, 2e-5, 4e-5)

for lr in "${lr_list[@]}" 
do
echo "${lr}"
export MODEL_DIR=en_atis_roberta
export MODEL_DIR=$MODEL_DIR"_"$lr"_"$SEED
python main.py --task atis \
                  --model_type bert \
                  --model_dir $MODEL_DIR \
                  --data_dir data \
                  --do_train \
                  --do_eval \
                  --num_train_epochs 30
done
