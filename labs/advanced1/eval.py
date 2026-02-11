from load_weights import load_pretrained_model
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import torch

model = load_pretrained_model()
model.eval()

# 快速验证
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


text = "The cat sat on the mat."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    print(inputs)
    outputs = model(x=inputs['input_ids'], padding_mask=inputs['attention_mask'])

    # 检查[CLS]向量（用于分类）
    cls_vector = outputs[1]
    print(f"[CLS]向量形状: {cls_vector.shape}")  # 应该是 [1, 768]
    print(f"[CLS]向量前5维: {cls_vector[0, :5]}")

    # 检查是否双向：对比第一个词和最后一个词的相似度
    hidden = outputs[0]
    first_token = hidden[0, 0]
    last_token = hidden[0, -2]  # [SEP]前一个是mat
    sim = F.cosine_similarity(first_token.unsqueeze(0), last_token.unsqueeze(0))
    print(f"'[CLS]'与'mat'的相似度: {sim.item():.4f} (应该较高，因为是双向)")
