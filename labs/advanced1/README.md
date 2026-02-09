# Advanced lab 1: Turning GPT into BERT

## Objective

In this assignment, you will take the existing from-scratch implementation of the GPT architecture from lab 2 and modify it to implement the BERT architecture with minimal necessary changes. You will validate your implementation by loading pre-trained BERT weights from [Hugging Face](https://huggingface.co) and verifying that it produces the same input-output behaviour as the official BERT model.

## Instructions

1. **Understand the Codebase**

   - Review the provided GPT implementation (`gpt2.py`), understanding its model components, training loop, and inference behaviour.
   - Identify key differences between GPT and BERT architectures.

2. **Modify GPT to implement BERT**

   - Make only the minimal necessary modifications to the existing GPT model to turn it into a BERT model.
   - Do not restructure the code unnecessarily — your goal is to adapt, not rewrite.

3. **Validate your implementation**

   - Load pre-trained BERT weights from Hugging Face for the [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) model.
   - Run random test inputs through both your BERT implementation and the Hugging Face model.
   - Compare the outputs to ensure your implementation behaves identically.

4. **Add your work to your portfolio**

   - Provide a `diff` file showing the differences between the original GPT implementation and your modified BERT implementation.
   - Include a short report summarising the changes you made and how you verified correctness.
   - Add the `diff` file and the report to your lab portfolio and present it at the oral exam.

## Hints & considerations

- Think about key architectural differences between GPT and BERT, such as causal vs. bidirectional attention and output layers.
- Use Hugging Face’s `AutoModel` and `AutoTokenizer` to compare outputs with a real BERT model.
- When validating, ensure that the tokenisation and preprocessing match between both models.

## Deliverables

- `gpt2bert.diff` – showing modifications from GPT to BERT
- `validate.py` – demonstrating how you verified the correctness of your implementation
- `report.md` – short report summarising your work

Good luck! 🚀
