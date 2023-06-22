import torch
import torch.nn as nn

def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out

class COTLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dim=3, rank=4):
        super(COTLinearLayer, self).__init__()

        # Define the reduction rate:
        self.r = rank
        eps = 1e-5

        assert in_features % self.r == 0, "in_features must be divisible by r"

        self.in_features=in_features
        self.out_features=out_features

        self.register_buffer('cross_attention_dim', torch.tensor(in_features))
        self.register_buffer('hidden_size', torch.tensor(out_features))

        self.fix_filt_shape = [in_features, out_features]

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
            with torch.no_grad():
                self.R.copy_(project(self.R, eps=self.eps))
            orth_rotate = self.cayley(self.R)
        else:
            with torch.no_grad():
                self.R.copy_(project_batch(self.R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)

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

        return out

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