import torch
import torch.nn as nn
import math
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
# non-causal linear attention
# https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-6)
    context = torch.einsum('...nd,...ne->...de', k, v)
    qkv = torch.einsum('...de,...nd->...ne', context, q)
    out = torch.einsum('...ne,...n->...ne', qkv, D_inv)
    return out
 


# ==================================RoPE Module=============================
# https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py


def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # xq: 64, 8, 64, 8
    # freqs_cis: 64, 8
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


# ======================Performer FAVOR+ Functions====================
def gram_schmidt(vectors):
    orthogonalized = []
    for v in vectors:
        for u in orthogonalized:
            v -= torch.dot(v, u) * u
        v = v / v.norm()  # Normalize to unit length
        orthogonalized.append(v)
    return torch.stack(orthogonalized)

# Generate Orthogonal Random Projections
def generate_orthogonal_random_projections(dim, m):
    random_matrix = torch.randn((m, dim), device=device)
    orthogonal_matrix = gram_schmidt(random_matrix)
    return orthogonal_matrix.T  # Return as (dim, m) for projection

# Phi+ kernel implementation
def phi_plus(z, m):
    batch_size, seq_len, dim = z.size()
    norm_squared = torch.norm(z, dim=-1, keepdim=True) ** 2
    orthogonal_random_matrix = generate_orthogonal_random_projections(dim, m)
    projected = z @ orthogonal_random_matrix  # [batch_size, seq_len, m]
    phi_plus_features = torch.exp(-norm_squared / 2) * torch.exp(projected) / torch.sqrt(torch.tensor(m, dtype=z.dtype, device=z.device))
    return phi_plus_features

# Phi++ kernel implementation
def phi_plus_plus(z, m):
    batch_size, seq_len, dim = z.size()
    norm_squared = torch.norm(z, dim=-1, keepdim=True) ** 2
    orthogonal_random_matrix = generate_orthogonal_random_projections(dim, m)
    projected = z @ orthogonal_random_matrix  # [batch_size, seq_len, m]
    phi_plus_plus_features = torch.exp(-norm_squared / 2) * torch.cat([
        torch.exp(projected),
        torch.exp(-projected)
    ], dim=-1) / torch.sqrt(torch.tensor(2 * m, dtype=z.dtype, device=z.device))
    return phi_plus_plus_features


# =====================Deterministic Linearizations======================
def relu_kernel(z):
    return torch.relu(z)


def exp_kernel(z):
    return torch.exp(z)


# ====================G Matrix Linerizations=========================
class StaticMatrix(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer('matrix', matrix)

    def forward(self, x):
        return x @ self.matrix

def random_hadamard_matrix(n):
    # Ensure n is a power of 2
    if (n & (n - 1)) != 0:
        raise ValueError("Dimension n must be a power of 2")

    # Base Hadamard matrix
    H = torch.tensor([[1.0]])

    # Recursive construction of Hadamard matrix
    while H.size(0) < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)

    # Random diagonal matrix for randomization
    D = torch.diag(torch.randint(0, 2, (n,)) * 2 - 1).float()  # Diagonal with +/-1

    # Normalize rows to have L2 norm 1
    H = H / (n ** 0.5)

    # Return randomized Hadamard matrix
    return D @ H

def combined_hadamard_matrices(n, num_blocks=3):
    """
    Create a product of multiple randomized Hadamard matrices.

    Args:
    - n (int): Dimension of the Hadamard matrix (must be a power of 2).
    - num_blocks (int): Number of Hadamard matrices to multiply (default is 3).

    Returns:
    - torch.Tensor: Combined randomized Hadamard matrix.
    """
    if (n & (n - 1)) != 0:
        raise ValueError("Dimension n must be a power of 2")

    # Initialize with identity matrix for multiplication
    combined = torch.eye(n)

    # Multiply multiple randomized Hadamard matrices
    for _ in range(num_blocks):
        combined = combined @ random_hadamard_matrix(n)

    return StaticMatrix(combined)

def givens_rotation_g(dim, num_rotations):
    # Define a Givens rotation wrapper
    class GivensRotation(nn.Module):
        def __init__(self, dim=dim, num_rotations=num_rotations):
            super().__init__()
            self.rotations = []
            for _ in range(num_rotations):
                a, b = torch.randint(0, dim, (2,))
                while a == b:
                    b = torch.randint(0, dim, (1,)).item()
                theta = torch.rand(1).item() * math.pi
                self.rotations.append((a, b, math.cos(theta), math.sin(theta)))

        def forward(self, x):
            for a, b, cos_theta, sin_theta in self.rotations:
                x_a, x_b = x[..., a].clone(), x[..., b].clone()
                x[..., a] = cos_theta * x_a - sin_theta * x_b
                x[..., b] = sin_theta * x_a + cos_theta * x_b
            return x
    return GivensRotation(dim, num_rotations)



def givens_rotation_g2(dim, num_rotations):
    """Faster implementation of Givens rotation matrix."""
    class GivensRotation(nn.Module):
        def __init__(self, dim, num_rotations):
            super().__init__()
            # Generate rotation parameters (a, b, theta) in batches
            self.a = torch.randint(0, dim, (num_rotations,))
            self.b = torch.randint(0, dim, (num_rotations,))
            # Ensure a != b
            while torch.any(self.a == self.b):
                self.b[self.a == self.b] = torch.randint(0, dim, (torch.sum(self.a == self.b).item(),))
            self.theta = torch.rand(num_rotations) * math.pi

        def forward(self, x):
            # Apply all rotations in a single batched operation
            for i in range(self.a.shape[0]):
                a, b, theta = self.a[i].item(), self.b[i].item(), self.theta[i].item()
                c, s = math.cos(theta), math.sin(theta)
                G = torch.eye(x.shape[-1], device=x.device)
                G[a, a], G[a, b], G[b, a], G[b, b] = c, -s, s, c
                x = x @ G

            return x
    return GivensRotation(dim, num_rotations)


def learnable_g(dim):
    return nn.Linear(dim, dim, bias=False).requires_grad_(True)

def fixed_random_g(dim):
    return StaticMatrix(torch.randn(dim, dim))

# Function for creating attention class of different flavours
def create_attention_class(kernel_func=None, g_func=None, rope_theta=10.0, apply_rope=True, rope_mixed=False):
    class RoPEAttention(Attention):
        def __init__(self, *args, rope_theta=rope_theta, apply_rope=apply_rope, rope_mixed=rope_mixed, **kwargs):
            super().__init__(*args, **kwargs)

            self.apply_rope = apply_rope
            self.rope_mixed = rope_mixed

            if g_func is not None:
                self.G = nn.ModuleList()
                for _ in range(self.num_heads):
                    self.G.append(g_func())

            if self.apply_rope:
                if self.rope_mixed:
                    self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)

                    freqs = init_2d_freqs(
                        dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta,
                        rotate=True
                    ).view(2, -1)
                    self.freqs = nn.Parameter(freqs, requires_grad=True)

                    t_x, t_y = init_t_xy(end_x=14, end_y=14)
                    self.register_buffer('freqs_t_x', t_x)
                    self.register_buffer('freqs_t_y', t_y)
                else:
                    self.compute_cis = partial(compute_axial_cis, dim=self.dim // self.num_heads, theta=rope_theta)
                    freqs_cis = self.compute_cis(end_x=14, end_y=14)
                    self.freqs_cis = freqs_cis

        def forward(self, x):
            B, N, C = x.shape

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            q, k, v = qkv[0], qkv[1], qkv[2]
            # B, heads, N, dim

            if self.apply_rope:
                ###### Apply rotary position embedding
                w = h = math.sqrt(x.shape[1] - 1)
                if self.rope_mixed:
                    t_x, t_y = self.freqs_t_x, self.freqs_t_y
                    if self.freqs_t_x.shape[0] != x.shape[1] - 1:
                        t_x, t_y = init_t_xy(end_x=w, end_y=h)
                        t_x, t_y = t_x.to(x.device), t_y.to(x.device)
                    freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
                else:
                    freqs_cis = self.freqs_cis
                    if self.freqs_cis.shape[0] != x.shape[1] - 1:
                        freqs_cis = self.compute_cis(end_x=w, end_y=h)
                    freqs_cis = freqs_cis.to(x.device)

                q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:].clone(), k[:, :, 1:].clone(), freqs_cis=freqs_cis)
                #########

            if kernel_func is not None:
                q = q.reshape(B * self.num_heads, N, -1)
                k = k.reshape(B * self.num_heads, N, -1)

                q = kernel_func(q)
                k = kernel_func(k)

                # reshape q and k back to original shape
                q = q.reshape(B, self.num_heads, N, -1)
                k = k.reshape(B, self.num_heads, N, -1)

                # multiply k, v
                x = linear_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
            elif g_func is not None:
                q = torch.cat([torch.relu(self.G[i](q[:, i, ...]).unsqueeze(1)) for i in range(self.num_heads)], dim=1)
                k = torch.cat([torch.relu(self.G[i](k[:, i, ...]).unsqueeze(1)) for i in range(self.num_heads)], dim=1)
                x = linear_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)

            return x
    return RoPEAttention