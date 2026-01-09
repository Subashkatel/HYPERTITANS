import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList

from einops import rearrange

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor
from hypll import nn as hnn

# functions

def l2norm(t):
    return F.normalize(t, dim = -1)

# norms

class LayerNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = self.gamma

        if gamma.ndim == 2:
            gamma = rearrange(gamma, 'b d -> b 1 d')

        return self.ln(x) * (gamma + 1.)

# norm + residual wrapper, as used in original TTT paper
# but could be removed

class ResidualNorm(Module):
    def __init__(
        self,
        dim,
        model: Module
    ):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.model = model

    def forward(self, x):

        out = self.model(x)

        return self.norm(out) + x

# memory mlp proposed in TTT

class MemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 2.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)

        self.weights = ParameterList([Parameter(torch.randn(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(
        self,
        x
    ):
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight

        return x

# memory mlp, but with gated residual + final projection

class GatedResidualMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 4.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, dim_hidden)),
                Parameter(torch.randn(dim_hidden, dim)),
                Parameter(torch.randn(dim * 2, dim)),
            ]) for _ in range(depth)
        ])

        self.final_proj = Parameter(torch.randn(dim, dim))

        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(
        self,
        x
    ):

        for weight1, weight2, to_gates in self.weights:
            res = x

            hidden = x @ weight1
            hidden = F.gelu(hidden)
            branch_out = hidden @ weight2

            # gated residual

            gates = cat((branch_out, res), dim = -1) @ to_gates
            x = res.lerp(branch_out, gates.sigmoid())

        return x @ self.final_proj

# memory mlp with factorized weights
# so can tradeoff capacity for smaller chunk sizes

class FactorizedMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        k = 32
    ):
        super().__init__()
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, k)),
                Parameter(torch.randn(k, dim)),
            ]) for _ in range(depth)
        ])

        for weight1, weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)

    def forward(
        self,
        x
    ):

        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight1 @ weight2

        return x

# an MLP modelled after the popular swiglu ff in modern transformers

class MemorySwiGluMLP(Module):
    def __init__(
        self,
        dim,
        depth = 1, # default to 2 layer MLP from TTT, depth of 2 would be 4 layer MLP, but done as 2 feedforwards with residual
        expansion_factor = 4.
    ):
        super().__init__()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        weights = []

        for _ in range(depth):
            weights.append(ParameterList([
                Parameter(torch.randn(dim, dim_inner * 2)),
                Parameter(torch.randn(dim_inner, dim)),
            ]))

        self.weights = ParameterList(weights)
        self.norm = LayerNorm(dim)

    def forward(self, x):

        for w1, w2 in self.weights:
            residual = x

            x, gates = (x @ w1).chunk(2, dim = -1)

            x = x * F.gelu(gates)

            x = x @ w2

            x = x + residual

        return self.norm(x)

# improvised attention as memory module

class MemoryAttention(Module):
    def __init__(
        self,
        dim,
        scale = 8.,
        expansion_factor = 2.
    ):
        super().__init__()
        self.scale = scale
        dim_ff_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            Parameter(torch.randn(dim, dim)), # queries
            Parameter(torch.randn(dim, dim)), # keys
            Parameter(torch.randn(dim, dim)), # values
            Parameter(torch.randn(dim, dim_ff_hidden)), # ff w1
            Parameter(torch.randn(dim_ff_hidden, dim)), # ff w2
        ])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):

        wq, wk, wv, ffw1, ffw2 = self.weights

        q = l2norm(x @ wq)
        k = l2norm(x @ wk)
        v = x @ wv

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,
            is_causal = True
        )

        # parallel attention + feedforward block
        # as in PaLM + Gpt-J

        h = F.gelu(x @ ffw1)
        ff_out = h @ ffw2

        return attn_out + ff_out

# hyperbolic memory MLP using Poincare ball model

class HyperbolicMemoryMLP(Module):
    """
    Memory MLP operating in hyperbolic (Poincare ball) space.

    Hyperbolic geometry is particularly suited for hierarchical data
    and can represent tree-like structures more efficiently than
    Euclidean space. This may benefit memory retrieval patterns
    that have hierarchical relationships.

    Reference: HypLL library (https://github.com/maxvanspengler/hyperbolic_learning_library)
    """
    def __init__(
        self,
        dim,
        depth = 2,
        expansion_factor = 2.,
        curvature = 1.0,
        learn_curvature = False
    ):
        super().__init__()

        # Poincare ball manifold
        self.manifold = PoincareBall(c=Curvature(value=curvature, requires_grad=learn_curvature))

        dim_hidden = int(dim * expansion_factor)

        # Build hyperbolic layers
        self.layers = ModuleList()
        dims = [dim] + [dim_hidden] * (depth - 1) + [dim]

        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.layers.append(hnn.HLinear(
                in_features=dim_in,
                out_features=dim_out,
                manifold=self.manifold
            ))
            # Add activation for all but the last layer
            if i < len(dims) - 2:
                self.layers.append(hnn.HReLU(manifold=self.manifold))

        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: Euclidean tensor of shape (batch, seq, dim) or (batch, dim)
        Returns:
            Euclidean tensor of same shape as input
        """
        orig_shape = x.shape

        # Flatten batch dimensions if needed
        if x.ndim > 2:
            x = rearrange(x, '... d -> (...) d')

        # Map from Euclidean to hyperbolic space via exponential map
        # TangentTensor wraps the Euclidean vector as a tangent vector at the origin
        tangent = TangentTensor(data=x, man_dim=-1, manifold=self.manifold)
        h = self.manifold.expmap(tangent)

        # Apply hyperbolic layers
        for layer in self.layers:
            h = layer(h)

        # Map back to Euclidean space via logarithmic map
        # logmap(x, y) maps from base point x to point y
        # Create origin on manifold by mapping zero tangent vector
        # Use multiplication by 0 to keep in computational graph
        zero_data = h.tensor * 0.0
        zero_tangent = TangentTensor(
            data=zero_data,
            man_dim=-1,
            manifold=self.manifold
        )
        origin = self.manifold.expmap(zero_tangent)
        out_tangent = self.manifold.logmap(origin, h)

        # Extract tensor with gradient tracking (use .tensor not .data)
        out = out_tangent.tensor

        # Restore original shape
        if len(orig_shape) > 2:
            out = out.view(*orig_shape)

        return out
