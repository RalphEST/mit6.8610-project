import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MLP(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hidden_dims: List[int], 
                 out_dim: int, 
                 p_dropout: float = 0, 
                 norm: str = None, 
                 actn: str = 'relu',
                 order: str = 'nd'):
        """
        Inputs:
        * `in_dim`:
        * `hidden_dims`:
        * `out_dim`:
        * `p_dropout`:
        * `norm`:
        * `actn`:
        * `order`:
        
        """
        super(MLP, self).__init__()
        
        actn2actfunc = {
            'relu': nn.ReLU(), 
            'leakyrelu': nn.LeakyReLU(), 
            'tanh': nn.Tanh(), 
            'sigmoid': nn.Sigmoid(), 
            'selu': nn.SELU(), 
            'softplus': nn.Softplus(), 
            'gelu': nn.GELU(),
            None: None,
            'None': None
        }
        norm2normlayer = {'bn': nn.BatchNorm1d(in_dim), 
                          'ln': nn.LayerNorm(in_dim), 
                          None: None, 
                          'None': None}
        
        assert p_dropout>=0 and p_dropout<=1, f"Invalid dropout probability {p_dropout}"
        assert actn in actn2actfunc.keys(), "Invalid activation function name"
        assert norm in norm2normlayer.keys(), "Invalid normalization option"
        assert order in ['nd', 'dn'], "Invalid normalization/dropout order"
        
        self.in_dim = in_dim
        self.hidden_hims = hidden_dims
        self.out_dim = out_dim
        
        actn = actn2actfunc[actn]
        norm = norm2normlayer[norm]
        
        if len(hidden_dims)>0:
            # input layer
            layers = [nn.Linear(self.in_dim, hidden_dims[0]), actn]

            # hidden layers
            for i in range(len(hidden_dims) - 1):
                layers += self.add_layer(
                    in_dim=hidden_dims[i], 
                    out_dim=hidden_dims[i+1], 
                    norm=norm, 
                    actn=actn, 
                    p_dropout=p_dropout, 
                    order=order
                )

            # output layer
            layers.append(nn.Linear(hidden_dims[-1], out_dim))
        else:
            layers = [nn.Linear(self.in_dim, self.out_dim)]
            layers += [actn] if actn else []

        self.fc = nn.Sequential(*layers)

    def add_layer(self,
                  in_dim: int,
                  out_dim: int,
                  norm: str,
                  actn: nn.Module,
                  p_dropout: float = 0.0,
                  order: str = 'nd'):
                
        # norm -> dropout or dropout -> norm
        if order == 'nd':
            layers = [norm] if norm else []
            layers += [nn.Dropout(p_dropout)] if p_dropout>0 else []
        elif order == 'dn':
            layers = [nn.Dropout(p_dropout)] if p_dropout>0 else []
            layers += [norm] if norm else []

        layers.append(nn.Linear(in_dim, out_dim))
        layers += [actn] if actn else []
        return layers

    def forward(self, x):
        output = self.fc(x)
        return output