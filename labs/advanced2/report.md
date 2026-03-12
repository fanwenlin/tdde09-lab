# Advanced Lab 2: Fine-tuning GPT-2 with LoRA - Report

## Design Decisions

Considering GPT model's causal attention, only rightmost token can aggregate the whole sentence's information. So I chose the next token after the end of sentence's hidden state as representation and use a linear layer for classification.

## Results

### Parameter Counts

| Model | Trainable Parameters | Accuracy |
|-------|---------------------|------------------|
| DistilBERT (full) | 66,955,010 | 91.4% |
| DistilBERT (LoRA) | 702722 | 88.6% |
| GPT-2 (full) | TODO | TODO |
| GPT-2 (LoRA) | 663552 | 91% |
