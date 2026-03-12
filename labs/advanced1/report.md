# Advanced lab 1: Turning GPT into BERT

## Differene between GPT and BERT
- [x] Static config: According to paper, BERT-base use a vocab_size of 30522 and a context window of 512
- [x] Attention Mask: GPT use causal mask to make it only looking the tokens on left side while BERT use a padding mask instead.
- [x] CLS: BERT has a special token \[CLS\] at the begining of sentence and there will be a CLS pool layer in the last encoder's output.
- [x] BERT use a smaller epsilon (1e-12) in Layer Normalization
- [x] BERT use Post-LayerNorm
- [x] Sentence Embedding: BERT has sentence seperator tokens and segmentation token embedding for Next Sentence Prediction task


## Validate
I compared the CLS embedding and pooler output of huggingface model and our model and get close results.

```sh
python3 validate.py
```

Result:
```
CLS equal: False
CLS close: True
CLS max diff: tensor(3.8147e-06)
POOL equal: False
POOL close: True
POOL max diff: tensor(1.5199e-06)
```
