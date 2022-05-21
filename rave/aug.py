import math
import torch
import torch.nn as nn

class AudioDA(nn.Module):
    def __init__(self, p=0.5, n_convs=6, rev=False, noise=True, mask=False):
        super().__init__()
        trans = []
        for i in range(n_convs):
            trans.append(RandomConv(p, 9, 1))
        trans.append(Reverse(p))
        trans.append(AddNoise(p, max_mix=0.3))
        trans.append(RandomMask(p, 0.3))
        self.trans = nn.Sequential(*trans)
    
    def forward(self, x):
        return self.trans(x)

class RandomApply(nn.Module):
    """
    Based on Training generative adversarial networks with limited data (Karras+, 2020)
    p should be < 0.8 so that it can be implicitly inverted
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, p=None):
        p = self.p if p is None else p
        if p < torch.rand(1):
            return x
        else:
            return torch.clamp(self.transform(x), -1, 1)

    def transform(self, x):
        raise NotImplementedError

class RandomMask(RandomApply):
    def __init__(self, p, max_mask_ratio=0.3):
        super().__init__(p=p)
        self.max_mask_ratio = max_mask_ratio

    def transform(self, x):
        mask_size = int(x.shape[-1] * self.max_mask_ratio * torch.rand(1))
        mask_start = torch.randint(0, x.shape[-1], (1,))
        mask_end = min(mask_start+mask_size, x.shape[-1])
        masked = x.clone()
        masked[..., mask_start:mask_end] = 0.0
        return masked

class AddNoise(RandomApply):
    def __init__(self, p, max_mix=0.2):
        super().__init__(p=p)
        self.max_mix = max_mix

    def transform(self, x):
        amp = self.max_mix * torch.rand(1, device=x.device) * x.abs().mean()
        return amp * (torch.rand_like(x)*2-1) + x

class RandomConv(RandomApply):
    # random non-causal filter
    def __init__(self, p, kernel_size=7, dilation=1):
        super().__init__(p=p)
        self.dilation = dilation
        self.register_buffer('kernel', (torch.rand(1, 1, kernel_size)*2-1)/math.sqrt(kernel_size), persistent=False)

    def transform(self, x):
        out = nn.functional.conv1d(x, self.kernel, bias=None, stride=1, padding='same', dilation=self.dilation)
        return out

class Reverse(RandomApply):
    def __init__(self, p):
        super().__init__(p=p)

    def transform(self, x: torch.Tensor):
        x = x.flip(-1)
        return x