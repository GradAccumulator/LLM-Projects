import torch
from models import Transformer

if __name__ == "__main__":
    model = Transformer(
        10,
        10,
        max_seq_len=10,
        num_kv_heads=5,
        embed_dim=10,
        ffn_dim=10,
    ) 
    x = torch.tensor([9,4,3]).reshape(3,1,1)
    out = [[0,0] for i in range(3)]
    model.eval()
    with torch.no_grad():
        with torch.inference_mode():
            for i in range(3):
                out[i][0] = model(x[i])
        for i in range(3):
            out[i][1] = model(x[i])

    for i in range(3):
        print(out[i][0])
        print(out[i][1])
        print('-'*100)