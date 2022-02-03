import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as sn


class Swish(nn.Module):
    def __init__(self, dim=-1):
        super(Swish, self).__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))

    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, :, None, None] * x)


class SmallMLP(nn.Module):
    def __init__(self, n_dims, n_out=1, n_hid=300, layer=nn.Linear, bias=True, dropout=.5, spectral_norm = True):
        super(SmallMLP, self).__init__()
        self._built = False
        if dropout is not None and not spectral_norm:
            self.net = nn.Sequential(
                layer(n_dims, n_hid, bias),
                Swish(n_hid),
                nn.Dropout(dropout),
                layer(n_hid, n_hid, bias),
                Swish(n_hid),
                nn.Dropout(dropout),
                layer(n_hid, n_out, bias)
            )
        elif dropout is not None and spectral_norm:
            self.net = nn.Sequential(
                sn(layer(n_dims, n_hid, bias=False)),
                Swish(n_hid),
                nn.Dropout(dropout),
                sn(layer(n_hid, n_hid, bias=False)),
                Swish(n_hid),
                nn.Dropout(dropout),
                sn(layer(n_hid, n_out, bias=False))
            )
        elif dropout is None and spectral_norm:
            self.net = nn.Sequential(
                sn(layer(n_dims, n_hid, bias=False)),
                Swish(n_hid),
                sn(layer(n_hid, n_hid, bias=False)),
                Swish(n_hid),
                sn(layer(n_hid, n_out, bias=False))
            )
        else:
            self.net = nn.Sequential(
                layer(n_dims, n_hid, bias),
                Swish(n_hid),
                layer(n_hid, n_hid, bias),
                Swish(n_hid),
                layer(n_hid, n_out, bias)
            )
        self.normalized = False

    def forward(self, x):
        x = x.view(x.size(0), -1)

        out = self.net(x)
        out = out.squeeze()

        if self.normalized:
            return out / (out.norm(dim=1, keepdim=True) + 1e-6)
        else:
            return out


# TODO: this does not parallel the SmallMLP at the moment
class LargeMLP(nn.Module):
    def __init__(self, n_dims, n_out=1, n_hid=300, layer=nn.Linear, dropout=False):
        super(LargeMLP, self).__init__()
        self._built = False
        if dropout:
            self.net = nn.Sequential(
                layer(n_dims, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_hid),
                Swish(n_hid),
                nn.Dropout(.5),
                layer(n_hid, n_out)
            )
        else:
            self.net = nn.Sequential(
                layer(n_dims, n_hid),
                Swish(n_hid),
                layer(n_hid, n_hid),
                Swish(n_hid),
                layer(n_hid, n_hid),
                Swish(n_hid),
                layer(n_hid, n_hid),
                Swish(n_hid),
                layer(n_hid, n_out)
            )
        self.normalized = False

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        out = out.squeeze()
        if self.normalized:
            return out / (out.norm(dim=1, keepdim=True) + 1e-6)
        else:
            return out
