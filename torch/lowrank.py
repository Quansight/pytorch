"""Implement various linear algebra algorithms for low rank matrices.
"""

import torch


def is_sparse(A):
    """Check if tensor A is a sparse tensor"""
    return isinstance(
        A, (torch.sparse.CharTensor, torch.sparse.ByteTensor,
            torch.sparse.ShortTensor, torch.sparse.IntTensor,
            torch.sparse.LongTensor, torch.sparse.HalfTensor,
            torch.sparse.BFloat16Tensor, torch.sparse.FloatTensor,
            torch.sparse.DoubleTensor))


def uniform(low=0.0, high=1.0, size=None, dtype=None, device=None):
    """Returns a tensor filled with random numbers from a uniform
    distribution on the interval :math:`[low, high)`

    Arguments::

      size (int...): a sequence of integers defining the shape of the
        output tensor.  Can be a variable number of arguments or a
        collection like a list or tuple.

      dtype (:class:`torch.dtype`, optional): the desired data type of
        returned tensor.  Default: if ``None``, uses a global default
        (see :func:`torch.set_default_tensor_type`).

      device (:class:`torch.device`, optional): the desired device of
        returned tensor.  Default: if ``None``, uses the current
        device for the default tensor type (see
        :func:`torch.set_default_tensor_type`). :attr:`device` will be
        the CPU for CPU tensor types and the current CUDA device for
        CUDA tensor types.

    """
    attrs = dict(dtype=dtype, device=device)
    if size is None:
        r = low + (high - low) * torch.rand(1, **attrs)[0]
    else:
        r = low + (high - low) * torch.rand(*size, **attrs)
    if dtype in [torch.complex32, torch.complex64, torch.complex128]:
        if size is None:
            i = low + (high - low) * torch.rand(1, **attrs)[0]
        else:
            i = low + (high - low) * torch.rand(*size, **attrs)
        return r + 1j * i
    return r

def get_floating_dtype(A):
    """Return the floating point dtype of tensor A.
    """
    return (A.flatten()[0] * 1.0).dtype


def conjugate(A):
    """Return conjugate of tensor A.

    .. note:: If A's dtype is not complex, A is returned.
    """
    if A.dtype in [torch.complex32, torch.complex64, torch.complex128]:    
        return A.conj()
    return A


def get_approximate_basis(A, q, niter=2):
    """Return tensor Q with q orthonormal columns such that :math:`Q Q^H
    A` approximates :math:`A`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al, 2009.

    .. note:: For an adequate approximation of a k-rank matrix A,
              where k is not known in advance but could be estimated,
              the number of Q columns q can be choosen according to
              the following criteria: in general, :math:`k <= q <
              2*k`. For large low-rank matrices, take :math:`q = k +
              5..10`.  If k is relatively small compared to
              :math:`min(m, n)`, choosing :math:`q = k + 0..2` may be
              sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Arguments::
        A (Tensor): the input tensor of size :math:`(m, n)`

        q (int): the dimension of subspace spanned by Q columns.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer. In most cases, the default
                               value 2 is more than enough.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """
    m, n = A.shape[-2:]

    dtype = get_floating_dtype(A)

    R = uniform(low=-1.0, high=1.0, size=(n, q),
                dtype=dtype, device=A.device)

    A_H = conjugate(A).t()
    (Q, _) = A.matmul(R).qr()
    for i in range(niter):
        (Q, _) = A_H.matmul(Q).qr()
        (Q, _) = A.matmul(Q).qr()
    return Q


def svd(A, q=6, niter=2):
    """Return the singular value decomposition ``(U, S, V)`` of a low-rank
    matrix A such that :math:`A \approx U diag(S) V^T`.

    .. note:: The implementation is based on the Algorithm 5.1 from
              Halko et al, 2009.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Arguments::
        A (Tensor): the input tensor of size :math:`(m, n)`

        q (int, optional): a slightly overestimated rank of A.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
   """
    m, n = A.shape[-2:]

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    if m < n:
        # computing the SVD approximation of a transpose in order to
        # keep B shape minimal
        Q = get_approximate_basis(A.t(), q, niter=niter)
        B_t = A.matmul(conjugate(Q))
        U, S, V = torch.svd(B_t)
        V = Q.matmul(V)
    else:
        Q = get_approximate_basis(A, q, niter=niter)
        B = conjugate(Q).t().matmul(A)
        U, S, V = torch.svd(B)
        U = Q.matmul(U)
    return U, S, V


def pca(A, q=None, center=True, niter=2):
    r"""Performs Principal Component Analysis (PCA) on a low-rank matrix.

    This function returns a namedtuple ``(U, S, V)`` which is the
    nearly optimal approximation of a singular value decomposition of
    a centered matrix :attr:`A` such that :math:`A = U diag(S) V^T`.

    .. note:: Different from the standard SVD, the size of returned
              tensors depend on the specified rank and q
              values as follows:
                - U is m x q matrix
                - S is q-vector
                - V is n x q matrix

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Arguments:

        A (Tensor): the input tensor of size :math:`(m, n)`

        q (int, optional): a slightly overestimated rank of A. By
                           default, q = min(6, m, n).

        center (bool, optional): if True, center the input tensor,
                                 otherwise, assume that the input is
                                 centered.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2

    References::

        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).

    """
    (m, n) = A.shape

    if q is None:
        q = min(6, m, n)
    elif not (q >=0 and q <= min(m, n)):
        raise ValueError('q(={}) must be non-negative integer'
                         ' and not greater than min(m, n)={}'
                         .format(q, min(m, n)))
    if not (niter >= 0):
        raise ValueError('niter(={}) must be non-negative integer'
                         .format(niter))

    dtype = get_floating_dtype(A)

    if center:
        ones_m1 = torch.ones((m, 1), dtype=dtype, device=A.device)
        c = A.sum(axis=-2) / m
        c = c.reshape((1, n))
        return pca(A - ones_m1.matmul(c), q=q, center=False, niter=niter)

    return svd(A, q, niter=niter)
