"""Locally Optimal Block Preconditioned Conjugate Gradient methods.
"""
# Author: Pearu Peterson
# Created: November 2019

import torch
from .lowrank import get_matmul, get_floating_dtype


class NumpyBackend(object):
    """Implements minimal torch API for numpy arrays.
    """
    def __init__(self):
        import numpy
        self.numpy = numpy
        self.float32 = numpy.float32
        self.float64 = self.double = numpy.float64

    def randn(self, size, dtype=None, device=None):
        return self.numpy.random.randn(*size).astype(dtype)

    def zeros(self, size, dtype=None, device=None):
        return self.numpy.zeros(size, dtype=dtype)

    def ones(self, size, dtype=None, device=None):
        return self.numpy.ones(size, dtype=dtype)

    def eye(self, size, dtype=None, device=None):
        return self.numpy.eye(size, dtype=dtype)

    def matmul(self, A, B):
        return get_matmul(A)(A, B)

    def chain_matmul(self, *matrices):
        if len(matrices) == 1:
            return matrices[0]
        elif len(matrices) == 2:
            A, B = matrices
        else:
            A, B = self.chain_matmul(*matrices[:-1]), matrices[-1]
        return get_matmul(A)(A, B)

    def cholesky(self, A, upper=False):
        if upper:
            L = self.numpy.linalg.cholesky(transpose(A))
            return transpose(L)
        L = self.numpy.linalg.cholesky(A)
        return L

    def inverse(self, A):
        return self.numpy.linalg.inv(A)

    def symeig(self, A, largest=False, eigenvectors=False):
        assert eigenvectors
        assert not largest
        return self.numpy.linalg.eigh(A)

    def diag_embed(self, A):
        if len(A.shape) == 1:
            return self.numpy.diag(A)
        raise NotImplementedError('{}.diag_embed for matrix with shape {}'
                                  .format(type(self).__name__, A.shape))

    def norm(self, A, dim=None):
        return self.numpy.linalg.norm(A, axis=dim)

    def numel(self, A):
        return A.nbytes / A.dtype.itemsize

    def where(self, A):
        return self.numpy.where(A)

    def qr(self, A, some=False):
        return self.numpy.linalg.qr(A,
                                    mode='reduced' if some else 'complete')

    def sparse_coo_tensor(self, indices, values, size, dtype=None, device=None):
        import scipy.sparse
        return scipy.sparse.coo_matrix((values, (indices[0], indices[1])), size, dtype=dtype)

    def arange(self, start, stop=None, dtype=None):
        if stop is None:
            stop = start
            start = 0
        return self.numpy.arange(start, stop, dtype=dtype)

    def sort(self, A, descending=False):
        if descending:
            return -self.numpy.sort(-A), NotImplemented
        return self.numpy.sort(A), NotImplemented

    def flip(self, A, dims=None):
        if dims == (-1, ):
            return A[..., ::-1]
        raise NotImplementedError('{}.flip for dims {}'
                                  .format(type(self).__name__, dims))


numpy_backend = NumpyBackend()


def get_backend(A):
    """Return array backend object that implements torch API for various
    matrix operations.
    """
    if isinstance(A, torch.Tensor):
        return torch
    return numpy_backend


def lobpcg(A, B=None, k=1, X=None, n=None, iK=None, niter=1000, tol=None,
           largest=True, tracker=None, method=None, **params):
    """Find the k smallest (or largest) eigenvalues and the corresponding
    eigenvectors of a symmetric positive defined generalized
    eigenvalue problem using matrix-free LOBPCG methods.

    This function is a front-end to the following LOBPCG algorithms
    selectable via `method` argument:

      `method="basic"` - the LOBPCG method introduced by Andrew
      Knyazev, see [Knyazev2001]. Supports dense, sparse, and batches
      of dense matrices. A less robust method, may fail when Cholesky
      is applied to singular input.

      `method="ortho"` - the LOBPCG method with orthogonal basis
      selection [StathopoulosEtal2002]. Supports dense and sparse
      matrices. A robust method.

    .. note:: In general, the basic method spends least time per
      iteration. However, the robust methods converge much faster and
      are more stable. So, the usage of the basic method is generally
      not recommended but there exist cases where the usage of the
      basic method may be preferred.

    Arguments:

      A (Tensor): the input tensor of size :math:`(*, m, m)`

      B (Tensor, optional): the input tensor of size :math:`(*, m,
                  m)`. When not specified, `B` is interpereted as
                  identity matrix.

      X (tensor, optional): the input tensor of size :math:`(*, m, n)`
                  where `k <= n <= m`. When specified, it is used as
                  initial approximation of eigenvectors. X must be a
                  dense tensor.

      iK (tensor, optional): the input tensor of size :math:`(*, m,
                  m)`. When specified, it will be used as preconditioner.

      k (integer, optional): the number of requested
                  eigenpairs. Default is `1`.

      n (integer, optional): if :math:`X` is not specified then `n`
                  specifies the size of the generated random
                  approximation of eigenvectors. Default value for `n`
                  is `k`. If :math:`X` is specifed, the value of `n`
                  is ignored.

      tol (float, optional): residual tolerance for stopping
                 criterion. Default is `feps ** 0.5` where `feps` is
                 smallest non-zero floating-point number of the given
                 input tensor `A` data type.

      largest (bool, optional): when True, solve the eigenproblem for
                 the largest eigenvalues. Otherwise, solve the
                 eigenproblem for smallest eigenvalues. Default is
                 `True`.

      method (str, optional): select LOBPCG method. See the
                 description of the function above. Default is
                 "ortho".

      niter (int, optional): maximum number of iterations. When
                 reached, the iteration process is hard-stopped and
                 the current approximation of eigenpairs is returned.
                 For infinite iteration but until convergence criteria
                 is met, use `-1`.

      tracker (callable, optional) : a function for tracing the
                 iteration process. When specified, it is called at
                 each iteration step with the following keyword
                 arguments:

                   `istep` - the current iteration step
                   `X` - the current approximation of eigenvectors
                   `E` - the current approximation of eigenvalues
                   `R` - the current residual
                   `nc` - the current number of converged eigenpairs
                   `rerr` - the current state of convergence criteria
                   `A`, `B`, `X` - the input matrices
                   `k`, `n`, `tol`, ... - the input parameters
                   `S`, `W`, `Z`, `np`, `ns` - various work arrays and
                     parameters of the iteration process

                 The recommended signature of tracker callable is
                 `tracker(istep, X, E, R, **params) -> None` where
                 `params` dictionary may contain other iteration
                 parameters that are specific to the selected
                 method. For instance, when `method` is `"basic"` then
                 the `params` dictionary contains:

                   `R_cond` - the condition number of Cholesky
                     factorization

                 When `method` is `ortho` then `params` contains:

                   `ortho.UMUmI_rerr`, `ortho.VMU_rerr`, `ortho.i`,
                   `ortho.j` - the statistics from orthogonalization
                   process.

                 Note that when `tracker` stores its arguments, it
                 must make copies of these.

                 Note that when `tracker` returns a non-zero value,
                 the iteration process will be interrupted and the
                 current approximation of the eigenpairs will be
                 returned as they are.

      params (dict, optional): various method-dependent parameters to
                 LOBPCG algorithms.

    Returns:

      E (Tensor): tensor of eigenvalues of size :math:`(*, k)`

      X (Tensor): tensor of eigenvectors of size :math:`(*, m, k)`

    References:

      [Knyazev2001] Andrew V. Knyazev. (2001) Toward the Optimal
      Preconditioned Eigensolver: Locally Optimal Block Preconditioned
      Conjugate Gradient Method. SIAM J. Sci. Comput., 23(2),
      517-541. (25 pages)
      `https://epubs.siam.org/doi/abs/10.1137/S1064827500366124`_

      [StathopoulosEtal2002] Andreas Stathopoulos and Kesheng
      Wu. (2002) A Block Orthogonalization Procedure with Constant
      Synchronization Requirements. SIAM J. Sci. Comput., 23(6),
      2165-2182. (18 pages)
      `https://epubs.siam.org/doi/10.1137/S1064827500370883`_

      [DuerschEtal2018] Jed A. Duersch, Meiyue Shao, Chao Yang, Ming
      Gu. (2018) A Robust and Efficient Implementation of LOBPCG.
      SIAM J. Sci. Comput., 40(5), C655-C676. (22 pages)
      `https://epubs.siam.org/doi/abs/10.1137/17M1129830`_

    """
    if B is not None:
        assert A.shape == B.shape, (A.shape, B.shape)

    if len(A.shape) > 2:
        if B is None:
            return batches_apply(
                lambda A_: lobpcg(A_, B=B, k=k, X=X, n=n, iK=iK, niter=niter,
                                  tol=tol, largest=largest, tracker=tracker,
                                  method=method, **params)
                , 2, (A,))
        return batches_apply(
            lambda A_, B_: lobpcg(A_, B=B_, k=k, X=X, n=n, iK=iK, niter=niter,
                                  tol=tol, largest=largest, tracker=tracker,
                                  method=method, **params)
            , 2, (A, B))
    torch = get_backend(A)

    dtype = get_floating_dtype(A)
    device = getattr(A, 'device', None)
    m = A.shape[-1]
    assert A.shape[-2] == m, A.shape  # expecting square matrix

    if X is None:
        if n is None:
            n = k
        X = torch.randn((m, n), device=device, dtype=dtype)
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
        tol = feps ** 0.5

    # Estimate A and B norms
    X_norm = norm(X)
    A_norm = norm(get_matmul(A)(A, X)) / X_norm
    B_norm = norm(get_matmul(B)(B, X)) / X_norm

    S = torch.zeros(A.shape[:-1] + (3 * n,), dtype=dtype, device=device)

    def converged_eigenpairs_count(R, X, E):
        # Use backward stable convergence criterion, see discussion in
        # Sec 4.3 of [DuerschEtal2018]
        rerr = torch.norm(R, dim=-2) / (torch.norm(X, dim=-2) * (A_norm + E[:X.shape[-1]] * B_norm))
        converged = rerr < tol
        count = 0
        for b in converged:
            if not b:
                # ignore convergence of following pairs to ensure
                # strict ordering of eigenpairs
                break
            count += 1
        return count, rerr

    if tracker is None:

        def tracker(*args, **kwargs):
            pass

    if method is None:
        method = 'ortho'

    return {'basic': lobpcg_worker_basic,
            'ortho': lobpcg_worker_ortho}[method](
                torch, A, B, X, S, m, n, k, iK, niter, tol, largest,
                tracker, converged_eigenpairs_count, **params)


def batches_apply(func, ndim, tensors):
    n = len(tensors[0].shape)
    if n > ndim:
        result = []
        results = None
        for items in zip(*[tensor.chunk(tensor.shape[0]) for tensor in tensors]):
            r = batches_apply(func, ndim, tuple(item[0] for item in items))
            if isinstance(r, tuple):
                if results is None:
                    results = [[] for r_ in r]
                else:
                    assert len(results) == len(r)
                for i, r_ in enumerate(r):
                    if isinstance(r_, int):
                        r_ = torch.tensor([r_], device=tensor.device)
                    else:
                        r_ = r_.reshape((1,) + r_.shape)
                    results[i].append(r_)
            else:
                result.append(torch.tensor([r], device=tensor.device))
        if results is not None:
            return tuple(map(torch.cat, results))
        return torch.cat(result)
    return func(*tensors)


def svqb(M, U, tau=1e-6, drop=False):
    """Return M-orthonormal U.

    .. note:: When `drop` is `False` then `svqb` is based on the
              Algorithm 4 from [DuerschPhD2015] that is a slight
              modification of the corresponding algorithm introduced
              in [StathopolousWu2002]. Otherwise,

    .. note:: `M=None` is iterpreted as identity matrix

    Arguments:

      M (Tensor) : positive definite matrix, size is (m, m)
      U (Tensor) : initial approximation, size is (m, n)
      tau (float, optional) : positive tolerance
      drop (bool, optional) : when True, drop columns that
                 contribution to the `span([U])` is small.

    Returns:

      U (Tensor) : M-orthonormal columns (:math:`U^T M U = I`), size
                   is (m, n1), where `n1 = n` if `drop` is `False,
                   otherwise `n1 <= n`.
    """
    torch = get_backend(U)
    if torch.numel(U) == 0:
        return U
    mm = torch.matmul
    mm_M = get_matmul(M)
    UMU = qform(M, U)
    d = UMU.diagonal(0, -2, -1)

    # detect and drop zero columns from U
    nz = torch.where(abs(d) != 0.0)
    if len(nz[0]) < len(d):
        return svqb(M, U[(Ellipsis,) + nz], tau=tau, drop=drop)

    D = torch.diag_embed(d ** -0.5)
    DUMUD = qform(UMU, D)
    E, Z = symeig(DUMUD, eigenvectors=True)
    t = tau * abs(E).max()
    if drop:
        keep = (Ellipsis,) + torch.where(E > t)
        E = E[keep]
        Z = Z[keep]
    else:
        E[torch.where(E < t)] = t
    return torch.chain_matmul(U, D, Z, torch.diag_embed(E ** -0.5))


def norm(A):
    """Return Frobenius norm of a real matrix.
    """
    if isinstance(A, torch.Tensor):
        try:
            return A.norm(dim=(-2, -1), p='fro')
        except RuntimeError:
            # e.g. conj is not available in CUDA
            return get_matmul(A)(transpose(A), A).trace() ** 0.5
    else:
        import numpy
        return numpy.linalg.norm(A, ord='fro')


def ortho(M, U, V, tau_ortho=1e-6, tau_drop=1e-6, tau_replace=1e-6,
          use_drop=False, **params):
    """Return M-orthonormal U that columns are M-orthogonal to V.

    .. note:: When `use_drop` is `False` then `ortho` is based on the
              Algorithm 3 from [DuerschPhD2015] that is a slight
              modification of the corresponding algorithm introduced
              in [StathopolousWu2002]. Otherwise, the function
              implements Algorithm 6 from [DuerschPhD2015]

    .. note:: If all U columns are M-collinear to V then the returned
              tensor U will be empty.

    Arguments:

      M (Tensor) : positive definite matrix, size is (m, m)
      U (Tensor) : initial approximation, size is (m, n)
      V (Tensor) : M-orthogonal external basis, size is (m, k)
      tau_ortho (float, optional) : positive tolerance for orthogonality checks.
      tau_drop (float, optional) : positive tolerance passed to svqb when called in drop mode
      tau_replace (float, optional) : positive tolerance passed to svqb when called in replace mode
      use_drop (bool, optional) : when True, enable dropping U columns
                   that have small contribution to the `span([U, V])`.

    Returns:

      U (Tensor) : M-orthonormal columns (:math:`U^T M U = I`) such
                   that :math:`V^T M U=0`, size is (m, n1), where `n1
                   = n` if `drop` is `False, otherwise `n1 <= n`.

      stats (dict) : statistics information
    """
    torch = get_backend(U)
    mm = torch.matmul
    mm_M = get_matmul(M)
    MV_norm = norm(mm_M(M, V))
    MU = mm_M(M, U)
    VMU = mm(transpose(V), MU)
    i = j = 0
    i_max = params.get('ortho.i_max', 3)
    j_max = params.get('ortho.j_max', 3)
    stats = {'ortho.UMUmI_rerr': {},
             'ortho.VMU_rerr': {}}
    for i in range(i_max):
        U = U - mm(V, VMU)
        drop = False
        tau_svqb = tau_drop
        for j in range(j_max):
            if use_drop:
                U = svqb(M, U, drop=drop, tau=tau_svqb)
                drop = True
                tau_svqb = tau_replace
            else:
                U = svqb(M, U, drop=False, tau=tau_replace)
            if torch.numel(U) == 0:
                # all initial U columns are M-collinear to V
                stats['ortho.i'] = i
                stats['ortho.j'] = j
                return U, stats
            MU = mm_M(M, U)
            UMU = mm(transpose(U), MU)
            U_norm = norm(U)
            MU_norm = norm(MU)
            R = UMU - torch.eye(UMU.shape[-1],
                                device=getattr(UMU, 'device', None),
                                dtype=UMU.dtype)
            R_norm = norm(R)
            rerr = R_norm / (MU_norm * U_norm)
            if rerr < tau_ortho:
                stats['ortho.UMUmI_rerr'][i, j] = rerr
                break
        VMU = mm(transpose(V), MU)
        VMU_norm = norm(VMU)
        rerr = VMU_norm / (MV_norm * U_norm)
        if rerr < tau_ortho:
            stats['ortho.VMU_rerr'][j] = rerr
            break
        if M is not None and M.shape[-1] < U.shape[-1] + V.shape[-1]:
            raise ValueError(
                'Overdetermined shape of U:'
                ' #M-cols(={}) >= #U-cols(={}) + #V-cols(={}) must hold'
                .format(M.shape[-1], U.shape[-1], V.shape[-1]))
    stats['ortho.i'] = i
    stats['ortho.j'] = j
    return U, stats


def get_RR_transform(B, S):
    """Return a transformation matrix that is used in Rayleigh-Ritz
    procedure for reducing a general eigenvalue problem :math:`(S^TAS)
    C = (S^TBS) C E` to a standard eigenvalue problem :math: `(Ri^T
    S^TAS Ri) Z = Z E` where `C = Ri Z`.

    Arguments:
      B (Tensor): the input matrix of size :math:`(*, m, m)`. When
                   specified as None, identity matrix is assumed.
      S (Tensor): the matrix basis for the search subspace, size is
                  :math:`(m, n)`.

    Returns:
      Ri (tensor): upper-triangular transformation matrix of size
                   :math:`(n, n)`.
    """
    torch = get_backend(S)
    mm = torch.matmul
    SBS = qform(B, S)
    d1 = SBS.diagonal(0, -2, -1) ** -0.5
    d = torch.zeros(d1.shape + (1, ),
                    device=getattr(d1, 'device', None),
                    dtype=d1.dtype)
    d[:, 0] = d1
    dd = mm(d, transpose(d))
    d_ = mm(d, transpose(torch.ones(d.shape,
                                    device=getattr(d, 'device', None),
                                    dtype=d.dtype)))
    R = torch.cholesky(dd * SBS, upper=True)
    # TODO: use LAPACK ?trtri as R is upper-triangular
    Rinv = torch.inverse(R)
    return Rinv * d_


def transpose(A):
    ndim = len(A.shape)
    return A.transpose(ndim - 1, ndim - 2)


def bform(X, A, Y):
    """Return bilinear form of matrices: :math:`X^T A Y`.
    """
    return get_matmul(X)(transpose(X), get_matmul(A)(A, Y))


def qform(A, S):
    """Return quadratic form :math:`S^T A S`.
    """
    return bform(S, A, S)


def residual(A, B, X, E):
    """Return residual :math:`A X - B X diag(E)`.
    """
    torch = get_backend(A)
    n = X.shape[-1]
    return get_matmul(A)(A, X) - torch.matmul(get_matmul(B)(B, X), torch.diag_embed(E[:n]))


def basis(A):
    """Return orthogonal basis of A columns.
    """
    torch = get_backend(A)
    try:
        Q = torch.orgqr(*torch.geqrf(A))
    except (RuntimeError, AttributeError):
        # torch.orgqr is not available in CUDA
        Q, _ = torch.qr(A, some=True)
    return Q


def symeig(A, largest=False, eigenvectors=True):
    """Return eigenpairs of A with specified ordering.
    """
    torch = get_backend(A)
    E, Z = torch.symeig(A, eigenvectors=eigenvectors)
    # assuming that E is ordered
    if largest:
        E = torch.flip(E, dims=(-1,))
        Z = torch.flip(Z, dims=(-1,))
    return E, Z


def lobpcg_worker_basic(torch, A, B, X, S, m, n, k, iK, niter, tol, largest,
                        tracker, converged_count, **params):
    mm = torch.matmul

    # Rayleigh-Ritz procedure, initialize
    Ri = get_RR_transform(B, X)
    R_diag_abs = abs(Ri.diagonal(0, -2, -1))
    R_cond = R_diag_abs.max() / R_diag_abs.min()

    M = qform(qform(A, X), Ri)
    E, Z = symeig(M, largest)
    X = mm(X, mm(Ri, Z))
    R = residual(A, B, X, E)
    np = 0
    nc, rerr = converged_count(R, X, E)
    S[..., :n] = X
    W = get_matmul(iK)(iK, R)
    ns = n + np + W.shape[-1]
    S[..., n + np:ns] = W

    tracker_args = dict(istep=0, X=X, E=E, R=R, S=S, W=W, k=k, n=n,
                        A=A, B=B, tol=tol, R_cond=R_cond,
                        nc=nc, np=np, ns=ns, Z=Z, rerr=rerr)
    if tracker(**tracker_args):
        niter = 0  # skip iteration

    while nc < k and niter:
        niter -= 1

        S_ = S[:, nc:ns]

        # Rayleigh-Ritz procedure
        Ri = get_RR_transform(B, S_)
        R_diag_abs = abs(Ri.diagonal(0, -2, -1))
        R_cond = R_diag_abs.max() / R_diag_abs.min()

        tracker_args['R_cond'] = R_cond
        M = qform(qform(A, S_), Ri)
        E_, Z = symeig(M, largest)

        # Update X, E, P
        X[:, nc:] = mm(S_, mm(Ri, Z[:, nc:n]))
        E[nc:] = E_[:n - nc]
        P = mm(S_, mm(Ri, Z[:, n:2 * n - nc]))
        np = P.shape[-1]

        # check convergence
        R = residual(A, B, X, E)
        prev_nc = nc
        nc, rerr = converged_count(R, X, E)
        if nc < prev_nc:
            raise RuntimeError('nc decreased: {} -> {}'.format(prev_nc, nc))

        # update S
        S[:, :n] = X
        S[:, n:n + np] = P
        W = get_matmul(iK)(iK, R[:, nc:])
        ns = n + np + W.shape[-1]
        S[:, n + np:ns] = W

        tracker_args['istep'] += 1
        tracker_args.update(
            nc=nc, np=np, ns=ns, rerr=rerr, R_cond=R_cond, R=R, W=W, Z=Z)
        if tracker(**tracker_args):
            break

    return E[:k], X[:, :k]


def lobpcg_worker_ortho(torch, A, B, X, S, m, n, k, iK, niter, tol, largest,
                        tracker, converged_count, **params):
    mm = torch.chain_matmul

    params['tol_ortho'] = params.get('tol_ortho', tol)
    params['tol_drop'] = params.get('tol_drop', tol)
    params['tol_replace'] = params.get('tol_replace', tol)
    params['ortho.i_max'] = params.get('ortho.i_max', 3)
    params['ortho.j_max'] = params.get('ortho.j_max', 3)

    # Rayleigh-Ritz procedure, initialize
    Ri = get_RR_transform(B, X)
    M = qform(qform(A, X), Ri)
    E, Z = symeig(M, largest)
    X = mm(X, Ri, Z)
    R = residual(A, B, X, E)
    np = 0
    nc, rerr = converged_count(R, X, E)
    S[:, :n] = X
    W, ortho_stats = ortho(B, R, X, **params)
    ns = n + np + W.shape[-1]
    S[:, n + np:ns] = W

    tracker_args = dict(istep=0, A=A, B=B, largest=largest,
                        X=X, E=E, R=R, S=S, W=W, k=k, n=n,
                        tol=tol,
                        nc=nc, np=np, ns=ns, Z=Z, rerr=rerr)
    tracker_args.update(ortho_stats)
    if tracker(**tracker_args):
        niter = 0  # skip iteration

    while nc < k and niter:
        niter -= 1

        S_ = S[:, nc:ns]

        # Rayleigh-Ritz procedure
        E_, Z = symeig(qform(A, S_), largest)

        # Update E, X, P
        X[:, nc:] = mm(S_, Z[:, :n - nc])
        E[nc:] = E_[:n - nc]
        P = mm(S_, Z[:, n - nc:], basis(transpose(Z[:n - nc, n - nc:])))
        np = P.shape[-1]

        # check convergence
        R = residual(A, B, X, E)
        prev_nc = nc
        nc, rerr = converged_count(R, X, E)
        if nc < prev_nc:
            raise RuntimeError('nc decreased: {} -> {}'.format(prev_nc, nc))

        # update S
        S[:, :n] = X
        S[:, n:n + np] = P
        W, ortho_stats = ortho(B, R[:, nc:], S[:, :n + np], **params)
        ns = n + np + W.shape[-1]
        S[:, n + np:ns] = W

        tracker_args['istep'] += 1
        tracker_args.update(
            ortho_stats, nc=nc, np=np, ns=ns, rerr=rerr, R=R, W=W, Z=Z)
        if tracker(**tracker_args):
            break

    return E[:k], X[:, :k]
