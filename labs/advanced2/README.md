# Advanced lab 2: Fine-tuning GPT-2 with LoRA

## Objective

In this assignment, you will combine what you have learned in the previous labs to fine-tune a GPT-2 model for sentiment classification using Low-Rank Adaptation (LoRA). Like in the first advanced lab, you are given minimal scaffolding: you will need to design, implement, and evaluate your own solution.

## Background

In labs 2 and 3, you implemented a GPT-2 model from scratch and wrote code to train it with a language modelling objective. In lab 4, you applied LoRA to a pre-trained DistilBERT model for binary sentiment classification on the IMDB movie review dataset. In this assignment, you will adapt the pretrained GPT-2 model for sequence classification and fine-tune it with LoRA.

## Instructions

1. **Set up the model**

   Load the pre-trained GPT-2 model from lab 3, or the original OpenAI model from lab 2. Replace the language modelling head with a classification head that maps the final hidden state to a binary label.

2. **Inject LoRA adapters**

   Implement LoRA adapters and inject them into the attention layers of the GPT-2 model, following the approach from lab 4. Only the LoRA adapter parameters and the classification head should be trainable; all other model parameters should be frozen.

3. **Fine-tune on IMDB**

   Fine-tune the adapted model on the same IMDB sentiment dataset used in lab 4. For tokenisation, you can use the `tiktoken` library with the `gpt-2` tokeniser.

4. **Evaluate and compare**

   Report the number of parameters of your models and the classification accuracy on the evaluation set. Compare your results against the following baselines from lab 4 in a short summary table:

   - DistilBERT (full)
   - DistilBERT (LoRA)
   - GPT-2 (full) [your work]
   - GPT-2 (LoRA) [your work]

5. **Add your work to your portfolio**

   Include a short report summarising the design decisions you made, the results you obtained, and your interpretation of the comparison between the models. Add the report and all code to your lab portfolio and present it at the oral exam.

## Hints & considerations

- GPT-2 uses causal (left-to-right) attention, unlike the bidirectional attention in BERT-family models. Think about what this means for how contextual information is aggregated across tokens when doing classification.
- The LoRA implementation from Lab 4 wraps a `nn.Linear` layer and can be reused here with minimal or no modification. The target layers and how you identify them will differ between GPT-2 and DistilBERT.
- GPT-2 uses a larger vocabulary and longer context than DistilBERT. You may want to truncate input sequences to a manageable length.

## Deliverables

- `gpt2-lora.py` — notebook containing your implementation, training run, and evaluation
- `report.md` — short report covering your design decisions, results, and discussion

Good luck! 🚀
