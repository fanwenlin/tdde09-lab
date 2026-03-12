from transformers import AutoTokenizer, AutoModel
import torch
from load_weights import load_pretrained_model




def load_tokenizer_and_hf_model():
    model_name = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModel.from_pretrained(model_name)
    hf_model.eval()
    return tokenizer, hf_model

def get_bert_output(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    if type(outputs) is tuple:
        last_hidden_states, pooler_output = outputs
    else:
        last_hidden_states = outputs.last_hidden_state
        pooler_output = outputs.pooler_output

    cls_embedding = last_hidden_states[:, 0, :]

    return cls_embedding, pooler_output[0]


if __name__ == "__main__":
    tok, hf_model = load_tokenizer_and_hf_model()
    my_model = load_pretrained_model()
    text = 'This is a sentence to test BERT model'
    inputs = tok(
        [text],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )
    c1, p1 = get_bert_output(hf_model, inputs)
    c2, p2 = get_bert_output(my_model, inputs)
    print("CLS equal:", torch.equal(c1,c2))
    print("CLS close:", torch.allclose(c1,c2,atol=1e-6))
    print("CLS max diff:", (c1-c2).abs().max())

    print("POOL equal:", torch.equal(p1,p2))
    print("POOL close:", torch.allclose(p1,p2,atol=1e-6))
    print("POOL max diff:", (p1-p2).abs().max())
