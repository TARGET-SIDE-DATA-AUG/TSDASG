# TARGET-SIDE-DATA-AUGMENTATION--FOR-SEQ-GENERATION

This is the source code and some evaluation scripts for our paper <TARGET-SIDEDATAAUGMENTATION  FORSEQUENCEGENERATION>.        

# TRAIN

## Machine Translation:

Setting for the WMT'14 EN->DE dataset:

Setting for the IWSLT'14 DE<->EN dataset:

2 ROUNDS:

```
fairseq-train DATA-BIN -a transformer_iwslt_de_en \
        --optimizer adam --lr 0.0005 -s de -t en --label-smoothing 0.1 --dropout 0.3 \
        --max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --max-update 30000 --warmup-updates 4000 \       
        --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' --save-dir SAVE-DIR  \        
        --share-all-embeddings --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \       
        --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples  --best-checkpoint-metric bleu \        
        --maximize-best-checkpoint-metric  --task translation --mixratio 0.4 --temperature 4  \       
        --activation-dropout 0.1 --attention-dropout 0.1  --log-format json --log-interval 50
```

3 ROUNS: 

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

## Dialog:

Setting for the Persona-Chat dataset:

```
fairseq-train DATA-BIN -a transformer \
        --optimizer adam --lr 0.0001 -s cxt -t res --label-smoothing 0.1 --dropout 0.3 \
        --max-tokens 4000 --min-lr '1e-09' \
        --criterion label_smoothed_cross_entropy --max-update 20000 --warmup-updates 3000 \       
        --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.999)' --save-dir SAVE-DIR  \        
        --share-all-embeddings --eval-bleu --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \       
        --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples  --best-checkpoint-metric bleu \        
        --maximize-best-checkpoint-metric  --task translation --mixratio 0.4 --temperature 4.5  \       
        --activation-dropout 0.1 --attention-dropout 0.1  --log-format json --log-interval 50        
```

Setting for the DailyDialog dataset:

```
fairseq-train DATA-BIN -a transformer \
        --share-all-embeddings     --optimizer adam --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0 --lr-scheduler inverse_sqrt     --warmup-init-lr 1e-07  \
        ---warmup-updates 4000 --lr 0.0005 --min-lr 1e-09  --weight-decay 0.0 \       
        --criterion label_smoothed_cross_entropy   --label-smoothing 0.1  --max-tokens 4096 \
        --update-freq 2 --no-progress-bar  --log-format json --max-update 200000  \
        --log-interval 10  --eval-bleu  --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu  \
        --seed 1111  --source-lang s --target-lang t --save-dir SAVE-DIR  \
        --temperature 4 --mixratio 0.5 \
        --attention-dropout 0.1 --activation-dropout 0.1 --log-format json --log-interval 10  
```


## Summarization:

## Other tasks:

Our method is universal so that you can use it on other sequence generation tasks and datasets. You can find details in 'Ablation' in our paper to know how to set the best values of 'temperature' and 'mixratio' of datasets you are using.

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

