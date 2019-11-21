"""Locally Optimal Block Preconditioned Conjugate Gradient methods


References:

  [1] Andrew V. Knyazev. (2001) Toward the Optimal Preconditioned Eigensolver:
  Locally Optimal Block Preconditioned Conjugate Gradient Method. SIAM
  J. Sci. Comput., 23(2), 517–541. (25 pages)
  https://epubs.siam.org/doi/abs/10.1137/S1064827500366124
  A report version: http://ccm.ucdenver.edu/reports/rep149.pdf

  [2] Jed A. Duersch, Meiyue Shao, Chao Yang, Ming Gu. (2018)  A Robust and
  Efficient Implementation of LOBPCG.  SIAM J. Sci. Comput., 40(5),
  C655–C676. (22 pages)
  https://epubs.siam.org/doi/abs/10.1137/17M1129830
  https://arxiv.org/pdf/1704.07458.pdf

"""

import torch

from .lowrank import get_matmul, batch_transpose, get_floating_dtype


def RayleighRitz(S, A, B=None, out=None):
    """Rayleigh-Ritz procedure - solve projected eigenvalue problem

    .. note:: The implementation is based on the Algorithm 2 from
              Duersch et al, 2018.

    .. note: input tensors :math:`A` and :math:`B` must contain
             symmetric positive definite matrices. By default,
             math:`B` is identity matrix.

    .. note: input tensor :math:`S` must contain matrices with
             linearly independent columns

    Arguments:

      S (Tensor): the matrix basis for the search subspace, size is
                  :math:`(*, m, n)`.

      A (Tensor): the input tensor of size :math:`(*, m, m)`
      B (Tensor, optional): the input tensor of size :math:`(*, m, m)`

    Returns:

      E (Tensor): tensor of eigenvalues of size :math:`(*, n)`
      C (Tensor): tensor of eigenvectors of size :math:`(*, n, n)`

    """
    mm = torch.matmul
    mm_S = get_matmul(S)
    mm_A = get_matmul(A)
    mm_B = get_matmul(B)

    S_T = batch_transpose(S)
    SBS = mm_S(S_T, mm_B(B, S))
    D = (SBS.diagonal(0, -2, -1) ** -0.5).diag_embed()
    DSBSD = mm(D, mm(SBS, D))    # n x n, TODO: optimize as D is diagonal
    R = torch.cholesky(DSBSD, upper=True)  # n x n
    iR = torch.inverse(R)        # TODO: use LAPACK ?trtri as R is upper-triangular
    DiR = mm(D, iR)
    SDiR = mm_S(S, DiR)
    X = mm(batch_transpose(SDiR), mm_A(A, SDiR))
    E, Z = torch.symeig(X, eigenvectors=True)
    if out is not None:
        return E, mm(DiR, Z, out=out)
    else:
        return E, mm(DiR, Z)


def lobpcg_basic(A, B=None, k=1, X=None, n=None, iK=None, niter=10, tol=None,
                 largest=False,
                 residual_collector=None, eigenvalue_collector=None):
    """Find the k smallest (or largest) eigenvalues and the corresponding
    eigenvectors of a symmetric positive defined generalized
    eigenvalue problem.

    .. note:: The implementation is based on the Algorithm 1 from
              Duersch et al, 2018.

    .. note: input tensor :math:`X` contains a initial approximation
             of eigenvectors.

    .. note: input tensors :math:`A` and :math:`B` must contain
             symmetric positive definite matrices.
             By default, math:`B` is identity matrix.

    .. note: input tensor :math:`iK` is an inverse of preconditioner matrix.

    Arguments:

      A (Tensor): the input tensor of size :math:`(*, m, m)`
      B (Tensor, optional): the input tensor of size :math:`(*, m, m)`

      X (tensor, optional): the input tensor of size :math:`(*, m, n)`
                  where `k <= n <= m`. When specified, X must be dense
                  tensor and its content will be overwritten.

      iK (tensor, optional): the input tensor of size :math:`(*, m, m)`

      n (integer, optional): if :math:`X` is not specified then `n`
                 specifies the size of the generated random
                 approximation of eigenvectors. Default value for `n`
                 is `k`. If :math:`X` is specifed, the value of `n` is
                 ignored.

      tol (float, optional): residual tolerance for stopping
                 criterion. By default, `tol = m * feps ** 0.5`.

      largest (bool, optional): when True, solve the eigenproblem for
                 largest eigenvalues.

      residual_collector (callable, optional): a single-argument
                 function that stores residuals (passed in to
                 collector function at each iteration step).

      eigenvalue_collector (callable, optional): a single-argument
                 function that stores eigenvalues (passed in to
                 collector function at each iteration step).

    Returns:
      E (Tensor): tensor of eigenvalues of size :math:`(*, k)`
      X (Tensor): tensor of eigenvectors of size :math:`(*, m, k)`

    """
    dtype = get_floating_dtype(A)
    device = A.device

    if largest:
        if B is None:
            B = torch.eye(*A.shape[-2:], **dict(dtype=dtype, device=device)).expand_as(A)
        E, X = lobpcg_basic(B, B=A, k=k, X=X, n=n, iK=iK, niter=niter, tol=tol,
                            largest=False, residual_collector=residual_collector)
        return 1 / E, X

    m = A.shape[-1]
    if X is None:
        if n is None:
            n = k
        X = torch.randn(A.shape[:-2] + (m, n), device=device, dtype=dtype)
    else:
        if X.shape[:-1] != A.shape[:-1]:
            raise ValueError('A and X have inconsistent shapes: {}[:-1] != {}[:-1]'
                             .format(A.shape, X.shape))
        n = X.shape[-1]

    if (m < 3 * n):
        raise ValueError(
            'LPBPCG algorithm is not applicable when the number of A rows (={})'
            ' is smaller than 3 x the number of requested eigenpairs (={})'
            .format(m, n))

    if tol is None:
        feps = {torch.float32: 1.2e-07,
                torch.float64: 2.23e-16}[dtype]
        tol = m * feps ** 0.5

    # matrix multiplication operators
    mm = torch.matmul
    mm_A = get_matmul(A)
    mm_B = get_matmul(B)        
    mm_iK = get_matmul(iK)

    # work storage S == [S0 S1 S2] == [S01 S2] == [S0 S12]
    S = torch.zeros(A.shape[:-2] + (m, 3 * n), dtype=dtype, device=device)
    S0 = S[..., :n]
    S1 = S[..., n:2 * n]
    S01 = S[..., :2 * n]
    S2 = S[..., 2 * n:]
    S12 = S[..., n:]

    for i in range(-1, niter):
        S0.copy_(X)                                    # S0 <- X

        if i == -1:
            S_ = S0
        elif i == 0:
            S_, S__ = S01, S1
        else:
            S_, S__ = S, S12

        # Rayleigh-Ritz procedure
        E, C = RayleighRitz(S_, A, B=B)

        if i == -1:
            C_ = C
        else:
            C_ = C[..., :n]

        # compute new approximation
        mm(S_, C_, out=X)                              # X <- S_ * C_

        # residual, note a typo in Algorithm 1
        R = mm_A(A, X) - mm_B(B, mm(X, E[..., :n].diag_embed()))

        # collect iteration information and check stoppage criterion
        res = R.norm(dim=-2)
        if residual_collector is not None:
            residual_collector(res)
        if eigenvalue_collector is not None:
            eigenvalue_collector(E[..., :k])
        if (res < tol).all():
            break

        # prepare S for the next interation
        if i >= 0:
            C__ = C[..., n:, :n]
            S2.copy_(mm(S__, C__))                     # S2 <- S__ * C__
        S1.copy_(mm_iK(iK, R))                         # S1 <- iK * R

    # return the requested number of eigenpairs
    return E[..., :k], X[..., :k]
