"""
Implementation of various resamplers, with attributes ``read``, ``paint`` and ``compensate``.

Reference
---------
https://arxiv.org/pdf/1512.07295
"""

from functools import partial

import numpy as np
import jax
from jax import numpy as jnp


def _compensate_tophat_convolution_kernel(order: int):

    def fn(value, kvec):
        kernel = 1.
        for kk in kvec:
            kernel *= jnp.sinc(kk / (2 * jnp.pi))
        return value * kernel**(-order)

    fn.kind = 'circular'

    return fn


def _aliasing_shotnoise_kernel(order: int):

    def kernel1(angle):  # NGP
        return 1.

    def kernel2(angle):  # CIC
        return 1. - 2. / 3. * jnp.sin(angle / 2.)**2

    def kernel3(angle):  # TSC
        s2 = jnp.sin(angle / 2.)**2
        return 1. - s2 + 2. / 15. * s2**2

    def kernel4(angle):  # PCS
        s2 = jnp.sin(angle / 2.)**2
        return 1. - 4. / 3. * s2 + 2. / 5. * s2 * s2 - 4. / 315. * s2**3

    _shotnoise_kernels = [None, kernel1, kernel2, kernel3, kernel4]

    def fn(value, kvec):
        kernel = 1.
        for kk in kvec:
            kernel *= _shotnoise_kernels[order](kk)
        return value * kernel

    fn.kind = 'circular'

    return fn


# Stolen from Hugo's montecosmo: https://github.com/hsimonfroy/montecosmo/blob/f4d318329a332d1a984e6a1fde6f5d59c4dd4336/montecosmo/nbody.py#L175


from itertools import product


# Elementwise: applied one axis at a time, so that the (N, ndim) arrays of separations and
# weights --- three quarters of the memory of painting and reading --- never exist.
_resampler_kernels_1d = [
    None,
    lambda s: jnp.ones_like(s), # NGP
    lambda s: 1 - s, # CIC
    lambda s: (s <= 1/2) * (3/4 - s**2) + (1/2 < s) / 2 * (3/2 - s)**2, # TSC
    lambda s: (s <= 1) / 6 * (4 - 6 * s**2 + 3 * s**3) + (1 < s) / 6 * (2 - s)**3, # PCS
]


def _index_weight(positions, id0, ishift, shape, order, idtype):
    """
    Return the flat (wrapped) mesh index and the resampling weight of each particle.

    Both are built axis by axis: the weight accumulates in a single (N,) array, and the index in a
    single (N,) integer, instead of the (N, ndim) separations, weights and indices that a
    vectorized form would materialize.
    """
    index, weight = None, None
    for axis in range(len(shape)):
        idx = id0[..., axis] + ishift[axis]
        s = jnp.abs(idx - positions[..., axis])
        w = _resampler_kernels_1d[order](s)
        weight = w if weight is None else weight * w
        idx = jnp.astype(idx % shape[axis], idtype)
        index = idx if index is None else index * jnp.astype(shape[axis], idtype) + idx
    return index, weight


def _get_index_dtype(size):
    """Integer type able to address a mesh of that many cells.

    Signed, and as narrow as the mesh allows: these index the gathers and scatters, so int32
    where it fits halves that memory. `np.min_scalar_type` is the obvious built-in but returns
    unsigned types, and `np.intp` is always 64-bit on a 64-bit host -- neither is usable here.
    """
    return np.int32 if size <= np.iinfo(np.int32).max else np.int64


def _paint_impl(mesh: tuple | jax.Array, positions, weights=1., order: int=2):
    """
    Paint the positions onto the mesh.
    If mesh is a tuple, paint on a zero mesh with such shape.
    """
    if isinstance(mesh, tuple):
        mesh = jnp.zeros(mesh)
    else:
        mesh = jnp.asarray(mesh)

    shape = np.asarray(mesh.shape, dtype='i8')
    idtype = _get_index_dtype(np.prod(shape))
    dtype = 'int16' # int16 -> +/- 32_767, should be enough
    id0 = (jnp.round if order % 2 else jnp.floor)(positions).astype(dtype)
    ishifts = np.arange(order) - (order - 1) // 2
    ishifts = np.array(list(product(* len(shape) * (ishifts,))), dtype=dtype)

    def step(carry, ishift):
        index, weight = _index_weight(positions, id0, ishift, shape, order, idtype)
        return carry.at[index].add(weights * weight), None

    # painted flat, such that the scatter takes a single index array rather than one per dimension
    mesh = jax.lax.scan(step, mesh.reshape(-1), ishifts)[0]
    return mesh.reshape(tuple(shape))


def _read_impl(mesh: jax.Array, positions, order: int=2, out=None):
    """Read the value at the positions from the mesh."""
    shape = np.asarray(mesh.shape, dtype='i8')
    idtype = _get_index_dtype(np.prod(shape))
    dtype = 'int16' # int16 -> +/- 32_767, should be enough
    id0 = (jnp.round if order % 2 else jnp.floor)(positions).astype(dtype)
    ishifts = np.arange(order) - (order - 1) // 2
    ishifts = np.array(list(product(* len(shape) * (ishifts,))), dtype=dtype)
    flat = mesh.reshape(-1)

    def step(carry, ishift):
        index, weight = _index_weight(positions, id0, ishift, shape, order, idtype)
        return carry + flat[index] * weight, None

    if out is None:
        out = jnp.zeros_like(positions, shape=positions.shape[:1])
    out = jax.lax.scan(step, out, ishifts)[0]
    return out


def _get_shifts(shape, order):
    ishifts = np.arange(order) - (order - 1) // 2
    return np.array(list(product(*len(shape) * (ishifts,))), dtype='int16')


def _get_id0(positions, order):
    dtype = 'int16'
    return (jnp.round if order % 2 else jnp.floor)(positions).astype(dtype)


def _kernel_derivative(order, s):
    """Derivative of the 1D resampling kernel with respect to ``s``."""
    if order == 1:  # NGP
        return jnp.zeros_like(s)
    if order == 2:  # CIC
        return jnp.where(s <= 1., -1., 0.)
    if order == 3:  # TSC
        return jnp.where(s <= 0.5, -2. * s,
                         jnp.where(s < 1.5, -(1.5 - s), 0.))
    if order == 4:  # PCS
        return jnp.where(s <= 1., -2. * s + 1.5 * s**2,
                         jnp.where(s < 2., -0.5 * (2. - s)**2, 0.))
    raise ValueError(f'unknown resampler order {order}')


def _index_weight_grad(positions, id0, ishift, shape, order, idtype):
    """Flat index, resampling weight and d(weight)/d(position)."""
    ws, dws = [], []
    index = None
    for axis in range(len(shape)):
        idx = id0[..., axis] + ishift[axis]
        diff = idx - positions[..., axis]
        s = jnp.abs(diff)
        w = _resampler_kernels_1d[order](s)
        dwds = _kernel_derivative(order, s)
        # id0 is piecewise constant, so d s / d position = -sign(idx - position).
        dwdpos = dwds * (-jnp.sign(diff))
        ws.append(w)
        dws.append(dwdpos)
        idx = jnp.astype(idx % shape[axis], idtype)
        index = idx if index is None else index * jnp.astype(shape[axis], idtype) + idx

    weight = ws[0]
    for w in ws[1:]:
        weight = weight * w
    dweight = []
    for axis in range(len(shape)):
        other = jnp.ones_like(ws[0])
        for j in range(len(shape)):
            if j != axis:
                other = other * ws[j]
        dweight.append(dws[axis] * other)
    return index, weight, jnp.stack(dweight, axis=-1)


def _make_paint_adjoint(order):
    @jax.custom_vjp
    def fn(mesh, positions, weights):
        return _paint_impl(mesh, positions, weights, order)

    def fwd(mesh, positions, weights):
        out = _paint_impl(mesh, positions, weights, order)
        return out, (positions, weights)

    def bwd(res, g):
        positions, weights = res
        # The cotangent has the mesh shape; keep it as a static Python tuple.
        shape = tuple(int(s) for s in g.shape)
        id0 = _get_id0(positions, order)
        ishifts = _get_shifts(shape, order)
        idtype = _get_index_dtype(np.prod(shape))
        g_flat = g.reshape(-1)
        scalar_weights = jnp.ndim(weights) == 0
        w = jnp.broadcast_to(weights, positions.shape[:1]) if scalar_weights else weights
        g_pos = jnp.zeros_like(positions)
        g_w = jnp.zeros(positions.shape[:1], dtype=jnp.result_type(w, positions))
        for ishift in ishifts:
            index, weight, dweight = _index_weight_grad(
                positions, id0, ishift, shape, order, idtype
            )
            g_at = g_flat[index]
            g_w = g_w + g_at * weight
            g_pos = g_pos + w[:, None] * g_at[:, None] * dweight
        if scalar_weights:
            g_w = jnp.sum(g_w)
        # output = input mesh + scatter, so the mesh cotangent is the identity
        return g, g_pos, g_w

    fn.defvjp(fwd, bwd)
    return fn


def _make_read_adjoint(order, shape):
    shape = tuple(int(s) for s in shape)

    @jax.custom_vjp
    def fn(mesh, positions, out):
        return _read_impl(mesh, positions, order, out)

    def fwd(mesh, positions, out):
        id0 = _get_id0(positions, order)
        ishifts = _get_shifts(shape, order)
        idtype = _get_index_dtype(np.prod(shape))
        flat = mesh.reshape(-1)
        values = []
        acc = out
        for ishift in ishifts:
            index, weight, _ = _index_weight_grad(
                positions, id0, ishift, shape, order, idtype
            )
            value = flat[index]
            values.append(value)
            acc = acc + value * weight
        # Storing the gathered values (N * nshift) is much smaller than the mesh.
        return acc, (positions, jnp.stack(values, axis=0))

    def bwd(res, g):
        positions, values = res
        id0 = _get_id0(positions, order)
        ishifts = _get_shifts(shape, order)
        idtype = _get_index_dtype(np.prod(shape))
        g_mesh = _paint_impl(
            jnp.zeros(shape, dtype=g.dtype), positions, g, order
        )
        g_pos = jnp.zeros_like(positions)
        for ishift_index, ishift in enumerate(ishifts):
            _, _, dweight = _index_weight_grad(
                positions, id0, ishift, shape, order, idtype
            )
            g_pos = g_pos + g[:, None] * values[ishift_index][:, None] * dweight
        # read output = out + gather, so the out cotangent is the identity
        return g_mesh, g_pos, g

    fn.defvjp(fwd, bwd)
    return fn


_PAINT_ADJOINTS = {order: _make_paint_adjoint(order) for order in range(1, 5)}
_READ_ADJOINT_CACHE = {}


def _get_read_adjoint(order, shape):
    key = (order, tuple(int(s) for s in shape))
    if key not in _READ_ADJOINT_CACHE:
        _READ_ADJOINT_CACHE[key] = _make_read_adjoint(order, key[1])
    return _READ_ADJOINT_CACHE[key]


def paint(mesh: tuple | jax.Array, positions, weights=1., order: int=2):
    """Paint positions onto the mesh, with a custom (paint/read) adjoint."""
    if isinstance(mesh, tuple):
        mesh = jnp.zeros(mesh)
    mesh = jnp.asarray(mesh)
    if weights is None:
        weights = jnp.ones(positions.shape[0], dtype=positions.dtype)
    return _PAINT_ADJOINTS[order](mesh, positions, weights)


def read(mesh: jax.Array, positions, order: int=2, out=None):
    """Read the mesh at positions, with a custom (read/paint) adjoint."""
    mesh = jnp.asarray(mesh)
    if out is None:
        out = jnp.zeros(positions.shape[:1], dtype=mesh.dtype)
    return _get_read_adjoint(order, mesh.shape)(mesh, positions, out)


# Define resampler namespaces, resampler.paint, resampler.read, resampler.compensate, resampler.aliasing_shotnoise, resampler.order
for i, name in enumerate(['ngp', 'cic', 'tsc', 'pcs']):
    order = i + 1
    globals()[name] = type(name, (), dict(paint=partial(paint, order=order), read=partial(read, order=order),
                                          compensate=_compensate_tophat_convolution_kernel(order),
                                          aliasing_shotnoise=_aliasing_shotnoise_kernel(order),
                                          order=order))


def get_resampler(resampler):
    return globals().get(resampler, resampler)