import torch

def softmax(
    input,
    dim=None
):
    #faz um shifting, impedindo que haja overflow
    shifted = input - input.max(
        dim=dim,
        keepdim=True
    ).values

    exp = torch.exp(shifted)

    return exp / exp.sum(
        dim=dim,
        keepdim=True
    )


#implementar o DROPOUT
