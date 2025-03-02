import torch
from torch import nn
from configs import HyperParameters

params = HyperParameters("slope_train_vit_simple")
device = torch.device(f"cuda:{params.gpu_index}" if torch.cuda.is_available() else "cpu")

class GBN(nn.Module):
    """
    Ghost Batch Normalization (GBN).
    Adapted from TabNet implementation:
    https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/tab_network.py
    """
    def __init__(self, inp, vbs=128, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs
    
    def forward(self, x):
        chunk = torch.chunk(x, max(1, x.size(0) // self.vbs), 0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)

class GLU(nn.Module):
    """ 
    Gated Linear Unit (GLU) activation 
    Adapted from TabNet implementation:
    https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/tab_network.py
    """
    def __init__(self, inp_dim, out_dim, fc=None, vbs=128):
        super().__init__()
        self.fc = fc if fc else nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        self.od = out_dim
    
    def forward(self, x):
        x = self.bn(self.fc(x))
        return x[:, :self.od] * torch.sigmoid(x[:, self.od:])

class FeatureTransformer(nn.Module):
    """ 
    Feature transformation block using GLU layers 
    Adapted from TabNet implementation:
    https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/tab_network.py
    """
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs=128):
        super().__init__()
        self.shared = nn.ModuleList([GLU(inp_dim, out_dim, shared[0], vbs=vbs)]) if shared else None
        self.independ = nn.ModuleList([GLU(out_dim, out_dim, vbs=vbs) for _ in range(n_ind)])
        self.scale = torch.sqrt(torch.tensor([.5], device=device))
    
    def forward(self, x):
        if self.shared:
            x = sum(glu(x) for glu in self.shared) * self.scale
        for glu in self.independ:
            x = (x + glu(x)) * self.scale
        return x

class AttentionTransformer(nn.Module):
    """ 
    Attention-based feature selection 
    Adapted from TabNet implementation:
    https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/tab_network.py
    """
    def __init__(self, inp_dim, out_dim, relax, vbs=128):
        super().__init__()
        self.fc = nn.Linear(inp_dim, out_dim)
        self.bn = GBN(out_dim, vbs=vbs)
        self.r = torch.tensor([relax], device=device)
    
    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = torch.sigmoid(a * priors)
        priors = priors * (self.r - mask)
        return mask

class DecisionStep(nn.Module):
    """ 
    Decision step combining feature transformation and attention 
    Adapted from TabNet implementation:
    https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/tab_network.py
    """
    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs=128):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs)
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)
    
    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        loss = (-mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, loss