import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Literal, List

def pad_sequence_side(
    sequences: torch.Tensor | List[torch.Tensor],
    batch_first = False,
    padding_value: float = 0,
    side: Literal['right', 'left'] = 'right'
):
    if side == 'right':
        return pad_sequence(
            sequences = sequences,
            batch_first = batch_first,
            padding_value = padding_value
        )
    else:
        return pad_sequence_left(
            sequences = sequences,
            batch_first = batch_first,
            padding_value = padding_value
        )

def pad_sequence_left(sequences, batch_first = False, padding_value = 0):

    max_size = sequences[0].size()  
    trailing_dims = max_size[1:]  
    max_len = max([s.size(0) for s in sequences])  
    if batch_first:  
        out_dims = (len(sequences), max_len) + trailing_dims  
    else:  
        out_dims = (max_len, len(sequences)) + trailing_dims  

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)  
    for i, tensor in enumerate(sequences):  
        length = tensor.size(0)  
        # use index notation to prevent duplicate references to the tensor  
        if batch_first:  
            out_tensor[i, -length:, ...] = tensor  
        else:  
            out_tensor[-length:, i, ...] = tensor  

    return out_tensor