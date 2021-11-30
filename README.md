# kr-dialect-machine-translation

- data dir
```
/nas/datahub/kr-dialect/train.csv
                        val.csv
                        test.csv
```

1. Train tokenizer(sentencepiece)

```
python preprocess.py --data-dir <path to data>
```
Result is as following.
```
data-dir/ bpe.model          # Tokenizer trained on train.csv
          bpe.train          
          bpe.vocab           
          vocab_dict.pickle  # Dictionary {idx : token}
```

2. Train Transformer Model
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--data-dir <path to data> \
--save-path <path to save model & log>
```

### TODO
- validation code in main.py
- Test code
