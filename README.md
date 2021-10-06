# TARGET-SIDE-DATA-AUGMENTATION--FOR-SEQ-GENERATION

# TRAIN

## Machine Translation:

For IWSLT'14 DE<->EN and WMT 14 EN->DE, you can 

## Dialog:

In our paper, we use DailyDialog and Persona-Chat. (We only use the self_original direction of Persona-Chat data in our experiments)

Setting for the WMT'14 EN->DE dataset:

Setting for the IWSLT'14 DE<->EN dataset:

2LUN

```

```

3LUN
```
fairseq-train DATA-BIN -a transformer_iwslt_de_en \

        --optimizer adam --lr 0.001 -s de -t en --label-smoothing 0.1 --dropout 0.3 \
        
        --max-tokens 4500 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.00006 \
        
        --criterion label_smoothed_cross_entropy --max-update 30000 --warmup-updates 3000 \
        
        --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' --save-dir SAVE-DIR  \
        
        --share-all-embeddings --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        
        --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples  --best-checkpoint-metric bleu \
        
        --maximize-best-checkpoint-metric  --task translation --mixratio 0.4 --temperature 4  \
        
        --activation-dropout 0.1 --attention-dropout 0.1  --log-format json --log-interval 50
```

## Summarization:

## OTHER TASKS:

Our method is universal so that you can use it on other sequence generation tasks and datasets. You can find details in 'Ablation' in our paper to know how to set the best hyper-parameters mentioned above of datasets you are using.

# EVALUATION 

## Machine Translation:

We use fairseq-generate command to evaluate BLEU score, using a command like this:

```
fairseq-generate DATA-BIN --path CHECKPOINT-PATH/checkpoint_best.pt --source-lang en --target-lang de \

--remove-bpe  --beam 5  --quiet
```

## Dialog: 

First of all, you should output the prediction of the test set to a file, using a command like this (Note that for dialog datasets, we use beam size 4 to generate):

```
fairseq-generate DATA-BIN --path CHECKPOINT-PATH/checkpoint_best.pt \

--source-lang s --target-lang t --remove-bpe  --beam 4  > FILE-NAME
```

Then, use 'scripts/compute_score.py' to compute its BLEU score, and use 'scripts/eval_nlg.py' to compute metrics like Met., CIDEr, and R-L. You should specify which file to compute like this:

```
python compute_score.py OUTPUT_FILE

python eval_nlg.py OUTPUT_FILE
```

## Summarization

