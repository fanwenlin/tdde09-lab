# Advanced lab 1: Turning GPT into BERT

## Differene between GPT and BERT
- [x] Static config: According to paper, BERT-base use a vocab_size of 30522 and a context window of 512
- [ ] Attention Mask: GPT use causal mask to make it only looking the tokens on left side while BERT use a padding mask instead.
- [x] CLS: BERT has a special token \[CLS\] at the begining of sentence and there will be a CLS pool layer in the last encoder's output.
- [x] BERT use a smaller epsilon (1e-12) in Layer Normalization
- [x] BERT use Post-LayerNorm
- [x] Sentence Embedding: BERT has sentence seperator tokens and segmentation token embedding for Next Sentence Prediction task


## Validate
