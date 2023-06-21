from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        #print('projecting !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    #if not mask.any():
    #    print("projecting batch!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out


class OPT_mlp(nn.Module):
    def __init__(self, in_features, out_features, bias=True, epsilon=None):
        super(OPT_mlp, self).__init__()
        # Get the number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        # Define the fixed Linear layer: v
        self.OPT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        # Initialize the weights
        # weight = torch.empty(out_features, in_features)
        # self.register_buffer('OPT_weight', weight)

        # Initialize the bias if needed
        # if bias:
        #     bias = torch.empty(out_features)
        #     self.register_buffer('OPT_bias', bias)
        # else:
        #     self.register_buffer('OPT_bias', None)


        #self.filt_shape = [in_features, in_features]
        self.fix_filt_shape = [in_features, out_features]

        # Define the trainable matrix parameter: R
        # Initialized as an identity matrix
        self.R_shape = [in_features, in_features]
        self.R = nn.Parameter(torch.eye(self.R_shape[0]), requires_grad=True)

        eps = 1e-2
        self.eps = eps * self.R_shape[0] * self.R_shape[0]

    def forward(self, x):
        with torch.no_grad():
            self.R.copy_(project(self.R, eps=self.eps))
        # R_updated = project(self.R, eps=self.eps)

        # Gram Schmidt parametrization
        orth_rotate = self.cayley(self.R)
        # orth_rotate = self.R
        # print('is orthogonal', self.is_orthogonal(orth_rotate))
        #print(self.R.shape)
        #sys.exit()
        # projected_R = project(self.R, eps=self.eps)
        # orth_rotate = self.cayley(projected_R)

        # fix filter
        fix_filt = self.OPT.weight.data
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(orth_rotate, fix_filt)
        filt = torch.transpose(filt, 0, 1)
  
        # Apply the trainable identity matrix
        bias_term = self.OPT.bias.data if self.OPT.bias is not None else None
        out = nn.functional.linear(input=x, weight=filt, bias=bias_term)
        # out = nn.functional.linear(input=x, weight=fix_filt.transpose(0, 1), bias=bias_term)

        return out 

    def cayley(self, data, jitter=1e-8):
        R = data
        r, c = list(R.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (R - R.t())
        I = torch.eye(r, device=R.device)
        # Add jitter to the diagonal elements of the matrix for numerical stability
        jitter_matrix = jitter * torch.eye(r, device=R.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I + skew, torch.inverse(I - skew + jitter_matrix))
        
        return Q

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)


class OPT_mlp_pe(nn.Module):
    def __init__(self, in_features, out_features, bias=True, epsilon=None, dim=3, r=4):
        super(OPT_mlp_pe, self).__init__()

        assert in_features % r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        # Define the fixed Linear layer: v
        self.OPT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        # Define the reduction rate:
        self.r = r

        #self.filt_shape = [in_features, in_features]
        self.fix_filt_shape = [in_features, out_features]

        eps = 1e-2
        # Define the trainable matrix parameter: R
        self.dim = dim
        if self.dim == 2:
            # Initialized as an identity matrix
            self.R_shape = [in_features // self.r, in_features // self.r]
            self.R = nn.Parameter(torch.zeros(self.R_shape[0], self.R_shape[0]), requires_grad=True)
  
            self.eps = eps * self.R_shape[0] * self.R_shape[0]
        else:
            # Initialized as an identity matrix
            self.R_shape = [self.r, in_features // self.r, in_features // self.r]
            R = torch.zeros(self.R_shape[1], self.R_shape[1])
            R = torch.stack([R] * self.r)
            self.R = nn.Parameter(R, requires_grad=True)
            self.eps = eps * self.R_shape[1] * self.R_shape[1]

    def forward(self, attn, x):
        orig_dtype = x.dtype
        dtype = self.R.dtype

        if self.dim == 2:
            #with torch.no_grad():
            #    self.R.copy_(project(self.R, eps=self.eps))
            orth_rotate = self.cayley(self.R)
        else:
            #with torch.no_grad():
            #    self.R.copy_(project_batch(self.R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.R)

        # Gram Schmidt parametrization
        # orth_rotate = self.cayley(self.R)
        # orth_rotate = self.R
        # print('is orthogonal', self.is_orthogonal(orth_rotate))
        # projected_R = project(self.R, eps=self.eps)
        # orth_rotate = self.cayley(projected_R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)
        # print('block_diagonal_matrix is identity', self.is_identity_matrix(block_diagonal_matrix))
        # print('block_diagonal_matrix is diagonal', self.is_orthogonal(block_diagonal_matrix))
        # fix filter
        fix_filt = attn.weight.data
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
        filt = torch.transpose(filt, 0, 1)
 
        # Apply the trainable identity matrix
        bias_term = attn.bias.data if attn.bias is not None else None
        if bias_term is not None:
            bias_term = bias_term.to(orig_dtype)

        out = nn.functional.linear(input=x.to(orig_dtype), weight=filt.to(orig_dtype), bias=bias_term)
        # out = nn.functional.linear(input=x, weight=fix_filt.transpose(0, 1), bias=bias_term)

        return out #.to(orig_dtype)

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I + skew, torch.inverse(I - skew))
        # Q = torch.mm(I - skew, torch.inverse(I + skew))
        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I + skew, torch.inverse(I - skew))

        return Q

    def block_diagonal(self, R):
        if self.dim == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]
            # for block in blocks:
            #     print('is diagonal 3', self.is_orthogonal(block))

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))


# feedforward
class GEGLU_opt(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        # self.proj = nn.Linear(dim_in, dim_out * 2)
        # self.proj = OPT_mlp(dim_in, dim_out * 2)
        self.proj = OPT_mlp_pe(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward_opt(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            # nn.Linear(dim, inner_dim),
            # OPT_mlp(dim, inner_dim),
            OPT_mlp_pe(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU_opt(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            # nn.Linear(inner_dim, dim_out)
            # OPT_mlp(inner_dim, dim_out),
            OPT_mlp_pe(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention_opt(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        # self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # self.to_q = OPT_mlp(query_dim, inner_dim, bias=False)
        self.to_q = OPT_mlp_pe(query_dim, inner_dim, bias=False)  

        # self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_k = OPT_mlp(context_dim, inner_dim, bias=False)
        self.to_k = OPT_mlp_pe(context_dim, inner_dim, bias=False)
        
        # self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_v = OPT_mlp(context_dim, inner_dim, bias=False)
        self.to_v = OPT_mlp_pe(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            # nn.Linear(inner_dim, query_dim),
            # OPT_mlp(inner_dim, query_dim),
            OPT_mlp_pe(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention_opt(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # self.to_q = OPT_mlp(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_k = OPT_mlp(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_v = OPT_mlp(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock_opt(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention_opt,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention_opt
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward_opt(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer_opt(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock_opt(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

