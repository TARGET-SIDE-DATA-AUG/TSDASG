# TARGET-SIDE-DATA-AUGMENTATION--FOR-SEQ-GENERATION

This is the source code and some evaluation scripts for our paper <TARGET-SIDEDATAAUGMENTATION  FORSEQUENCEGENERATION>.        

Our code is based on https://github.com/pytorch/fairseq.
        
Note that in this repo, we set 'beta' = 1 and 'iteration' = 1, while you can specify 'temperature' and 'alpha(fixed)' in commands. (all details about these hyper-parameters can be found in section 4.4 in our paper). Besides, you can modify our code to use schedule alpha, more iterations, and different beta. We will show how to modify our code in the end of this file.

# COMMANDS
## TRAIN

### Machine Translation:

Setting for the WMT'14 EN->DE dataset:

```
fairseq-train DATA-BIN \
        -a transformer --optimizer adam --lr 0.001 \
        -s en -t de --label-smoothing 0.1 \
        --dropout 0.1 --relu-dropout 0.1 --attention-dropout 0.1 \
        --max-tokens 4096 --update-freq 16 \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt \
        --criterion label_smoothed_cross_entropy --max-update 500000 --warmup-updates 4000 \
        --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' \
        --save-dir ckpt --share-all-embeddings \
        --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses --eval-bleu-remove-bpe \
        --valid-subset "valid,test" \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --seed 1 --alpha 0.5 \
        --no-avg-loss --temperature 2 --fp16
```
        
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
        --maximize-best-checkpoint-metric  --task translation --alpha 0.4 --temperature 4  \       
        --activation-dropout 0.1 --attention-dropout 0.1  --log-format json --log-interval 50
```

3 ROUNDS: 

```
fairseq-train DATA-BIN -a transformer_iwslt_de_en \
        --optimizer adam --lr 0.001 -s de -t en --label-smoothing 0.1 --dropout 0.3 \
        --max-tokens 4500 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.00006 \
        --criterion label_smoothed_cross_entropy --max-update 30000 --warmup-updates 3000 \       
        --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' --save-dir SAVE-DIR  \        
        --share-all-embeddings --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \       
        --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples  --best-checkpoint-metric bleu \        
        --maximize-best-checkpoint-metric  --task translation --alpha 0.4 --temperature 4  \       
        --activation-dropout 0.1 --attention-dropout 0.1  --log-format json --log-interval 50
```

### Dialog:

Setting for the Persona-Chat dataset:

```
fairseq-train DATA-BIN -a transformer \
        --optimizer adam --lr 0.0001 -s cxt -t res --label-smoothing 0.1 --dropout 0.3 \
        --max-tokens 4000 --min-lr '1e-09' \
        --criterion label_smoothed_cross_entropy --max-update 20000 --warmup-updates 3000 \       
        --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.999)' --save-dir SAVE-DIR  \        
        --share-all-embeddings --eval-bleu --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \       
        --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples  --best-checkpoint-metric bleu \        
        --maximize-best-checkpoint-metric  --task translation --alpha 0.4 --temperature 4.5  \       
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
        --temperature 4 --alpha 0.5 \
        --attention-dropout 0.1 --activation-dropout 0.1 --log-format json --log-interval 10  
```


### Summarization:

### Other tasks:

Our method is universal so that you can use it on other sequence generation tasks and datasets. You can find details in 'Ablation' in our paper to know how to set the best values of 'temperature' and 'alpha' of datasets you are using.

## EVALUATION 

### Machine Translation:

We use fairseq-generate command to evaluate BLEU score, using a command like this:

```
fairseq-generate DATA-BIN --path CHECKPOINT-PATH/checkpoint_best.pt \
        --source-lang en --target-lang de \
        --remove-bpe  --beam 5  --quiet
```

### Dialog: 

First of all, you should output the prediction of the test set to a file, using a command like this (Note that for dialog datasets, we use beam size 4 to generate):

```
fairseq-generate DATA-BIN --path CHECKPOINT-PATH/checkpoint_best.pt \

        --source-lang s --target-lang t --remove-bpe \
        --beam 4  > FILE-NAME
```

Then, use 'scripts/compute_score.py' to compute its BLEU score, and use 'scripts/eval_nlg.py' to compute metrics like Met., CIDEr, and R-L. You should specify which file to compute like this:

```
python compute_score.py OUTPUT_FILE

python eval_nlg.py OUTPUT_FILE
```

### Summarization

Same as above.
        
# MODIFICATION

Bigger or smaller beta, schedule alpha, or more iterations aren't that good, so we didn't leave any args in commands to modify them. However, if you want to figure out their effects, you can follow our guide below to extend our code.        
        
## Beta

In /fairseq/fairseq/criterions/label_smoothed_cross_entropy.py, line 132:

```python
     loss_total_without_kd = loss2 * mix_ratio + loss * (1 - mix_ratio) + dk       
```
        
Here, 'beta' stands for the coefficient of 'dk'. Empirically, 'beta' = 1 gives the best performance. You can change this line to try different 'beta', such as '2 * dk', which means 'beta' is 2.
        
## Schedule alpha

Empirically, fixed alpha is good in our method, but you can design your own schedule alpha function and use it.

In /fairseq/fairseq_cli/train.py, line 137, please add your schedule alpha fuction here. For example:
        
```python
    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch :

        mix_ratio = 1.5 / np.sqrt(np.log(epoch) + 1)    # Write your fuction here. It will overwrite the fixed alpha.
                                                            
        .....
                                                                   
                                                               
```

When you have added your fuction here, you can ignore hyper-parameter '--alpha' in commands.

                                                                   
## More iterations

We will show you how to use 2 iterations, and you can follow the guide to extend more.                                                                   
                                                                   
Start with /fairseq/fairseq/models/transformer.py, line 285:                                                               

```python
                                                                   
        decoder_out2 = None
        decoder_out3 = None # new added
                                                                   
        if self.training and mix_ratio > 0 and mix_ratio < 1:
           
            # lines below are original.
                                                          
            x = decoder_out1[0].clone()
            length = len(x[0])
            for idx in range(length - 1, -1, -1):
                x[:,idx] = x[:,idx - 1]     
            
            x[:,0, 2] = 2 * torch.max(x[:,0])   
            x = utils.softmax(x / self.temperature, dim = 2)
            
            with torch.no_grad():     
                embed_matrix = self.decoder.embed_tokens.weight.clone()   # vocab_size * embed_lenghth (10152 * 512)        
                decoder_in = torch.einsum('blv,ve->ble', x, embed_matrix) # batch * len * embed_lenghth
         
            #second round decoder
            decoder_out2 = self.decoder(
                    prev_output_tokens,          
                    encoder_out=encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                    mix_ratio=mix_ratio,
                    x=decoder_in,
                )
            
                                                             
            #lines below are new added.
                                                             
            x2 = decoder_out2[0].clone()
            length = len(x2[0])
            for idx in range(length - 1, -1, -1):
                x2[:,idx] = x2[:,idx - 1]     
            
            x2[:,0, 2] = 2 * torch.max(x2[:,0])   
            x2 = utils.softmax(x2 / self.temperature, dim = 2)
            
            with torch.no_grad():     
                embed_matrix = self.decoder.embed_tokens.weight.clone()   # vocab_size * embed_lenghth (10152 * 512)        
                decoder_in2 = torch.einsum('blv,ve->ble', x2, embed_matrix) # batch * len * embed_lenghth
         
            #second round decoder
            decoder_out3 = self.decoder(
                    prev_output_tokens,          
                    encoder_out=encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                    mix_ratio=mix_ratio,
                    x=decoder_in2,
                )
        
         # note that here we add an extra return value                                                    
        return decoder_out1, decoder_out2, decoder_out3                                                    
```

                                                             
Besides, you should modify /fairseq/fairseq/criterions/label_smoothed_cross_entropy.py, line 68. The number of return values should match with iterations you use.
Here, cause we add an extra iteration, we should add a return value and name it as net_output3. We also need to pass this new parameter to fuction compute_loss, which is in line 71.
                                                             
```python
        net_output1, net_output2, net_output3 = model(**sample["net_input"], mix_ratio=mix_ratio)       

        loss, nll_loss = self.compute_loss(model, net_output1, net_output2, net_output3, sample, mix_ratio, reduce=reduce)
```
                                                             
In line 100, don't forget add a parameter in function prototype of compute_loss:
            
                                                             
```python            
         def compute_loss(self, model, net_output1, net_output2, net_output3, sample, mix_ratio, reduce=True):
```
            
In this fuction, copy the code of the first iteration, and paste it in the end. Then, just modify some variables' names. Don't forget modify the loss fuction by the way.
            
