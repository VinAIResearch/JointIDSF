# Intent detection and slot filling for Vietnamese



Details of our dataset construction, JointIDSF model architecture, and experimental results can be found in our [following paper](https://arxiv.org/abs/2104.02021):

    @article{jointidsf,
    title     = {{Intent detection and slot filling for Vietnamese}},
    author    = {Mai Hoang Dao, Thinh Hung Truong, Dat Quoc Nguyen},
    booktitle = {},
    year      = {2021}
    }

**Please CITE** our paper whenever our dataset or model implementation is used to help produce published results or incorporated into other software.

## Installation
- Python version >= 3.6; PyTorch version >= 1.4.0
```
    git clone https://github.com/VinAIResearch/JointIDSF.git
    cd JointIDSF/
    pip install -r requirements.txt
```


## Training and Evaluation
To reproduce the results in the paper please run
```
    run_phobert_crf.sh
    run_phobert_crf_attn.sh
    run_xlmr_crf.sh
    run_xlmr_crf_attn.sh
```
Example usage:
```
python3 main.py --task word-level \
                  --model_type phobert \
                  --model_dir ./phobert_crf/ \
                  --data_dir data \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 50 \
                  --tuning_metric mean_intent_slot \
                  --use_crf \
                  --token_level word \
                  --embedding_type soft \
                  --intent_loss_coef 0.3 \
                  --learning_rate 3e-5
```