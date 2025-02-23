import math

import torch, tntorch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter


__all__ = [
    'Linear'
]



class Linear(nn.Module):
    r"""Meta linear moudule.
    Uses tntorch library to compress the high dimensional meta weights.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        order: The order of meta. Default: 1.
        depth: number of iterations that input will go through this module.
        kwargs: arguments passed to tntorch.Tensor constructor.

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
        order: The order of meta. Default: 1.
        depth: number of iterations that input will go through this module.

    Examples::

        >>> m = Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        order: int = 1,
        depth: int = 1,
        device=None,
        dtype=None,
        **kwargs
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order # meta param
        self.depth = depth # meta param
        self.weight = Parameter(tntorch.rand((out_features, in_features) + (in_features,) * order, **kwargs, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        self.weight = Parameter(tntorch.rand((out_features, in_features) + (in_features,) * order, **kwargs, **factory_kwargs))
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        sqrt_d = math.sqrt(x.shape[-1])
        for i in range(self.depth):
            # first perform meta forward
            w = self.weights
            for j in range(self.order):
                w = torch.matmul(x, w.T) / sqrt_d
            x = F.linear(x, w, self.bias)
        return x

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, order={self.order}, iters={self.depth}"
