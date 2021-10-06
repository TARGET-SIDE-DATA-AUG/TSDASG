# TARGET-SIDE-DATA-AUGMENTATION--FOR-SEQ-GENERATION



## EVALUATION 

# Machine Translation:

For IWSLT'14 DE<->EN and WMT 14 EN->DE, We use fairseq-generate command to evaluate BLEU score, using a command like this:

```
fairseq-generate DATA-BIN --path CHECKPOINT-PATH/checkpoint_best.pt --source-lang en --target-lang de --remove-bpe  --beam 5 --quiet
```

# Dialog: 

First of all, you should output the prediction of the test set to a file, using a command like this (Note that for dialog datasets, we use beam size 4 to generate):

fairseq-generate DATA-BIN --path CHECKPOINT-PATH/checkpoint_best.pt --source-lang s --target-lang t --remove-bpe  --beam 4  > FILE-NAME

Then, use 'scripts/compute_score.py' to compute its BLEU score, and use 'scripts/eval_nlg.py' to compute metrics like Met., CIDEr, and R-L. You should specify which file to compute like this:

```
python compute_score.py OUTPUT_FILE

python eval_nlg.py OUTPUT_FILE
```

# Summarization

