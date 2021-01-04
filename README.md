# JointIDSF: English - Vietnamese Joint Intent Detection - Slot Filling

Based on the code from https://github.com/monologg/JointBERT

## Model implemented:
- JointBERT, JointMBERT, JointDistilBERT, JointAlBERT
- JointRoBERTa, JointXLM-R, JointPhoBERT

## Added Early Stopping

## Tuning metrics to choose:
- Validation loss
- Intent accuracy
- Slot F-1
- Semantic Frame Accuracy (Both intent and slots need to be correct)

## Run

```
sh run_{model_type}.sh
model_type: bert, mbert, roberta, xlmr, phobert (only works for Vietnamese)
```

