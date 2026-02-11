from gpt2 import Model, Config

def load_weights(model: Model, pretrained):
    with torch.no_grad():

        model.wte.weight.copy_(pretrained['bert.embeddings.word_embeddings.weight'])
        model.wpe.weight.copy_(pretrained['bert.embeddings.position_embeddings.weight'])
        model.type_token_wpe.weight.copy_(pretrained['bert.embeddings.token_type_embeddings.weight'])


        # attn
        for i in range(model.config.n_layer):
            prefix = f'bert.encoder.layer.{i}.'

            q_w = pretrained[f'{prefix}attention.self.query.weight']
            k_w = pretrained[f'{prefix}attention.self.key.weight']
            v_w = pretrained[f'{prefix}attention.self.value.weight']
            model.h[i].attn.c_attn.weight.copy_(
                torch.cat([q_w, k_w, v_w], dim=0)
            )

            q_b = pretrained[f'{prefix}attention.self.query.bias']
            k_b = pretrained[f'{prefix}attention.self.key.bias']
            v_b = pretrained[f'{prefix}attention.self.value.bias']
            model.h[i].attn.c_attn.bias.copy_(
                torch.cat([q_b, k_b, v_b], dim=0)
            )

            model.h[i].attn.c_proj.weight.copy_( pretrained[f'{prefix}attention.output.dense.weight'] )
            model.h[i].attn.c_proj.bias.copy_( pretrained[f'{prefix}attention.output.dense.bias'] )
            model.h[i].ln_1.g.copy_( pretrained[f'{prefix}attention.output.LayerNorm.gamma'] )
            model.h[i].ln_1.b.copy_( pretrained[f'{prefix}attention.output.LayerNorm.beta'] )

            model.h[i].mlp.c_fc.weight.copy_( pretrained[f'{prefix}intermediate.dense.weight'] )
            model.h[i].mlp.c_fc.bias.copy_( pretrained[f'{prefix}intermediate.dense.bias'] )
            model.h[i].mlp.c_proj.weight.copy_( pretrained[f'{prefix}output.dense.weight'] )
            model.h[i].mlp.c_proj.bias.copy_( pretrained[f'{prefix}output.dense.bias'] )



            model.h[i].ln_2.g.copy_( pretrained[f'{prefix}output.LayerNorm.gamma'] )
            model.h[i].ln_2.b.copy_( pretrained[f'{prefix}output.LayerNorm.beta'] )

        # lnf
        # cls_pooler
        model.ln_f.g.copy_(pretrained['bert.embeddings.LayerNorm.gamma'])
        model.ln_f.b.copy_(pretrained['bert.embeddings.LayerNorm.beta'])
        model.cls_pooler.weight.copy_(pretrained['bert.pooler.dense.weight'])
        model.cls_pooler.bias.copy_(pretrained['bert.pooler.dense.bias'])
    return model


def download_model():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="google-bert/bert-base-uncased",
        filename="pytorch_model.bin"
    )

    print(path)

import torch
def load_model_dict(path='./pytorch_model.bin'):
    state_dict = torch.load(path)
    return state_dict

def load_pretrained_model(path='./pytorch_model.bin'):
    return load_weights(Model(Config()), load_model_dict())

if __name__ == "__main__":
    # download_model()
    d = load_model_dict()
    # print(d.keys())
    load_weights(Model(Config()), d)
