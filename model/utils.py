import torch
from torch import nn
from configs import HyperParameters

params = HyperParameters("slope_train_vit_simple")
device = torch.device(f"cuda:{params.gpu_index}" if torch.cuda.is_available() else "cpu")

class UnlearnableIdentityConvModel(nn.Module):
    def __init__(self):
        super(UnlearnableIdentityConvModel, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=64,      # Number of input channels
            out_channels=64,     # Number of output channels
            kernel_size=1,       # 1x1 convolution
            stride=1,            # Stride of 1
            padding=0,           # No padding
            bias=False           # No bias term needed
        )
        
        # Initialize weights to be identity
        self.conv.weight.data = torch.eye(64).view(64, 64, 1, 1)
        
        # Freeze the weights
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

class Sparsemax(nn.Module):
    """
    This class is adapted from the TabNet implementation:
    https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/tab_network.py

    Sparsemax activation function for neural networks.
    """
    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)
        
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, device=device,step=1, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]
        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)
        self.output = torch.max(torch.zeros_like(input), input - taus)
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)
        return output
    def backward(self, grad_output):
        dim = 1
        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))
        return self.grad_input   
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
        self.smax = Sparsemax()
        self.r = torch.tensor([relax], device=device)
    
    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = self.smax(a*priors) # mask = torch.sigmoid(a * priors)
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