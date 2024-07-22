import math

import numpy as np
import torch
from scipy.linalg import hadamard


def get_greatest_common_factor(N):
    # 获取最大公因数
    sqrt = int(np.sqrt(N))
    for i in range(sqrt, 0, -1):
        if N % i == 0:
            return i, N // i


def get_orthogonal_matrix(size, mode, dtype=torch.float32, device='cpu'):
    if mode == 'random':
        return random_orthogonal_matrix(size, dtype, device)
    elif mode == 'hadamard':
        assert is_pow2(size), f"{size} is not power of two"
        coefficient = math.sqrt(2) ** math.log2(size)
        matrix = hadamard(n=size, dtype=np.float32)
        return torch.from_numpy(matrix).to(dtype=dtype, device=device) / coefficient
    else:
        raise ValueError(f'Unknown mode {mode}')


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def random_orthogonal_matrix(size, dtype=torch.float32, device='cpu'):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float32, device=device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q.to(dtype=dtype)
