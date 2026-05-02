"""
Gaussian mutual information estimation.

Uses the formula: I(X;Y) = 0.5 * (log|Σ_X| + log|Σ_Y| - log|Σ_XY|)
where Σ_XY is the joint covariance matrix.
"""

import torch


def safe_logdet_batch(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Batched log determinant with numerical stabilization.
    
    Args:
        M: (B, D, D) batch of matrices
        
    Returns:
        (B,) log determinants
    """
    B, D, _ = M.shape
    I = torch.eye(D, device=M.device, dtype=M.dtype).unsqueeze(0).expand(B, D, D)

    for i in range(5):
        try:
            sign, logabs = torch.linalg.slogdet(M)
            if (sign > 0).all() and torch.isfinite(logabs).all():
                return logabs
        except:
            pass
        M = M + eps * (10 ** i) * I
    
    return torch.linalg.slogdet(M + 1e-3 * I)[1]


def mutual_information_gaussian(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    eps: float = 1e-3
) -> torch.Tensor:
    """
    Compute MI assuming Gaussian distributions using Schur complement.
    
    Args:
        X: (B, N, D) batch of samples
        Y: (B, N, D) batch of samples
        
    Returns:
        (B,) MI values
    """
    B, N, D = X.shape
    
    # center
    X = X - X.mean(dim=1, keepdim=True)
    Y = Y - Y.mean(dim=1, keepdim=True)

    # covariances
    I_reg = eps * torch.eye(D, device=X.device).unsqueeze(0)
    XT_X = torch.bmm(X.transpose(1, 2), X) / (N - 1) + I_reg
    YT_Y = torch.bmm(Y.transpose(1, 2), Y) / (N - 1) + I_reg
    XT_Y = torch.bmm(X.transpose(1, 2), Y) / (N - 1)

    # schur complement: Σ_X|Y = Σ_X - Σ_XY Σ_Y^{-1} Σ_YX
    Z = torch.linalg.solve(YT_Y, XT_Y.transpose(1, 2))
    schur = XT_X - torch.bmm(XT_Y, Z)
    schur = 0.5 * (schur + schur.transpose(1, 2))  # symmetrize

    log_X = safe_logdet_batch(XT_X)
    log_schur = safe_logdet_batch(schur)

    # I(X;Y) = 0.5 * (log|Σ_X| - log|Σ_X|Y|)
    return 0.5 * (log_X - log_schur)
