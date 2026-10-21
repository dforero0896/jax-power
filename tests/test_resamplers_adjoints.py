"""Tests for the custom paint/read adjoints in ``jaxpower.resamplers``.

The custom VJPs are compared against the plain autodiff path through
``_paint_impl`` / ``_read_impl`` on random small meshes.
"""

import numpy as np
import jax
import jax.numpy as jnp

from jaxpower import MeshAttrs, ParticleField, resamplers


SHAPES = [(6, 5, 4), (7, 6)]
ORDERS = (1, 2, 3, 4)
ATOL = 2e-5
RTOL = 1e-4


def _problem(seed=0, shape=(6, 5, 4), n=400):
    rng = np.random.default_rng(seed)
    shape = tuple(shape)
    ndim = len(shape)
    low = 0.1 * np.ones(ndim)
    high = np.asarray(shape) - 0.1
    positions = jnp.asarray(rng.uniform(low, high, size=(n, ndim)), dtype=jnp.float32)
    weights = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float32)
    mesh = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    out = jnp.asarray(rng.normal(size=(n,)), dtype=jnp.float32)
    return mesh, positions, weights, out


def _assert_close(test, ref, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(np.asarray(test), np.asarray(ref), atol=atol, rtol=rtol)


def test_forward_parity():
    for shape in SHAPES:
        mesh, positions, weights, out = _problem(shape=shape)
        for order in ORDERS:
            _assert_close(
                resamplers.paint(mesh, positions, weights, order=order),
                resamplers._paint_impl(mesh, positions, weights, order),
            )
            _assert_close(
                resamplers.paint(shape, positions, weights, order=order),
                resamplers._paint_impl(shape, positions, weights, order),
            )
            _assert_close(
                resamplers.read(mesh, positions, order=order),
                resamplers._read_impl(mesh, positions, order),
            )
            _assert_close(
                resamplers.read(mesh, positions, order=order, out=out),
                resamplers._read_impl(mesh, positions, order, out),
            )
    print('forward parity: ok')


def test_paint_gradients():
    for shape in SHAPES:
        mesh, positions, weights, _ = _problem(shape=shape, seed=1)
        for order in ORDERS:
            def loss_custom(mesh, positions, weights):
                return jnp.sum(
                    resamplers.paint(mesh, positions, weights, order=order) ** 2
                )

            def loss_reference(mesh, positions, weights):
                return jnp.sum(
                    resamplers._paint_impl(mesh, positions, weights, order) ** 2
                )

            grads_custom = jax.grad(loss_custom, argnums=(0, 1, 2))(
                mesh, positions, weights
            )
            grads_reference = jax.grad(loss_reference, argnums=(0, 1, 2))(
                mesh, positions, weights
            )
            for custom, reference in zip(grads_custom, grads_reference):
                _assert_close(custom, reference)
                assert np.isfinite(np.asarray(custom)).all()
    print('paint gradients: ok')


def test_read_gradients():
    for shape in SHAPES:
        mesh, positions, _, out = _problem(shape=shape, seed=2)
        for order in ORDERS:
            def loss_custom(mesh, positions, out):
                return jnp.sum(
                    resamplers.read(mesh, positions, order=order, out=out) ** 2
                )

            def loss_reference(mesh, positions, out):
                return jnp.sum(
                    resamplers._read_impl(mesh, positions, order, out) ** 2
                )

            grads_custom = jax.grad(loss_custom, argnums=(0, 1, 2))(
                mesh, positions, out
            )
            grads_reference = jax.grad(loss_reference, argnums=(0, 1, 2))(
                mesh, positions, out
            )
            for custom, reference in zip(grads_custom, grads_reference):
                _assert_close(custom, reference)
                assert np.isfinite(np.asarray(custom)).all()
    print('read gradients: ok')


def test_scalar_weights():
    for shape in SHAPES:
        mesh, positions, _, _ = _problem(shape=shape, seed=3)
        for order in ORDERS:
            def loss_custom(weights):
                return jnp.sum(
                    resamplers.paint(shape, positions, weights, order=order) ** 2
                )

            def loss_reference(weights):
                return jnp.sum(
                    resamplers._paint_impl(shape, positions, weights, order) ** 2
                )

            custom = jax.grad(loss_custom)(1.0)
            reference = jax.grad(loss_reference)(1.0)
            _assert_close(custom, reference, atol=2e-3, rtol=2e-3)
    print('scalar weights: ok')


def test_jit_gradients():
    shape = (6, 5, 4)
    mesh, positions, weights, _ = _problem(shape=shape, seed=4)
    order = 3

    def loss_custom(mesh, positions, weights):
        return jnp.sum(resamplers.paint(mesh, positions, weights, order=order) ** 2)

    def loss_reference(mesh, positions, weights):
        return jnp.sum(resamplers._paint_impl(mesh, positions, weights, order) ** 2)

    grads_custom = jax.jit(jax.grad(loss_custom, argnums=(0, 1, 2)))(
        mesh, positions, weights
    )
    grads_reference = jax.grad(loss_reference, argnums=(0, 1, 2))(
        mesh, positions, weights
    )
    for custom, reference in zip(grads_custom, grads_reference):
        _assert_close(custom, reference)
    print('jit gradients: ok')


def test_particlefield_paint_gradient():
    rng = np.random.default_rng(5)
    mattrs = MeshAttrs(meshsize=8, boxsize=100.0, fft_backend='jax')
    n = 400
    positions = jnp.asarray(rng.uniform(-50.0, 50.0, size=(n, 3)), dtype=jnp.float32)
    weights = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float32)
    particles = ParticleField(positions=positions, weights=weights, attrs=mattrs)

    cell_positions = (
        positions + mattrs.boxsize / 2.0 - mattrs.boxcenter
    ) / mattrs.cellsize

    for resampler in ('ngp', 'cic', 'tsc', 'pcs'):
        mesh = particles.paint(resampler=resampler, compensate=False, interlacing=0)
        reference = getattr(resamplers, resampler).paint(
            jnp.zeros(tuple(mattrs.meshsize)), cell_positions, weights
        )
        _assert_close(mesh.value, reference)

        def loss_particles(positions):
            return jnp.sum(
                particles.clone(positions=positions)
                .paint(resampler=resampler, compensate=False, interlacing=0)
                .value ** 2
            )

        def loss_reference(cell_positions):
            return jnp.sum(
                getattr(resamplers, resampler).paint(
                    jnp.zeros(tuple(mattrs.meshsize)), cell_positions, weights
                ) ** 2
            )

        grad_particles = jax.grad(loss_particles)(positions)
        grad_reference = jax.grad(loss_reference)(cell_positions) / mattrs.cellsize
        _assert_close(grad_particles, grad_reference)
        assert np.isfinite(np.asarray(grad_particles)).all()
    print('ParticleField.paint gradient: ok')


def test_realmeshfield_read_gradient():
    rng = np.random.default_rng(6)
    mattrs = MeshAttrs(meshsize=8, boxsize=100.0, fft_backend='jax')
    n = 400
    positions = jnp.asarray(rng.uniform(-50.0, 50.0, size=(n, 3)), dtype=jnp.float32)
    mesh = mattrs.create(
        kind='real', fill=jnp.asarray(rng.normal(size=(8, 8, 8)), dtype=jnp.float32)
    )
    cell_positions = (
        positions + mattrs.boxsize / 2.0 - mattrs.boxcenter
    ) / mattrs.cellsize

    for resampler in ('ngp', 'cic', 'tsc', 'pcs'):
        values = mesh.read(positions, resampler=resampler, compensate=False)
        reference = getattr(resamplers, resampler).read(mesh.value, cell_positions)
        _assert_close(values, reference)

        def loss_mesh(positions):
            return jnp.sum(mesh.read(positions, resampler=resampler, compensate=False) ** 2)

        def loss_reference(cell_positions):
            return jnp.sum(
                getattr(resamplers, resampler).read(mesh.value, cell_positions) ** 2
            )

        grad_mesh = jax.grad(loss_mesh)(positions)
        grad_reference = jax.grad(loss_reference)(cell_positions) / mattrs.cellsize
        _assert_close(grad_mesh, grad_reference)
        assert np.isfinite(np.asarray(grad_mesh)).all()
    print('RealMeshField.read gradient: ok')


def test_interlaced_compensated_gradient():
    from functools import partial

    rng = np.random.default_rng(7)
    mattrs = MeshAttrs(meshsize=12, boxsize=100.0, fft_backend='jax')
    n = 200
    positions = jnp.asarray(rng.uniform(-50.0, 50.0, size=(n, 3)), dtype=jnp.float32)
    weights = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float32)
    particles = ParticleField(positions=positions, weights=weights, attrs=mattrs)

    for resampler, order in (('ngp', 1), ('cic', 2), ('tsc', 3), ('pcs', 4)):
        def loss(positions, weights):
            return jnp.sum(
                particles.clone(positions=positions, weights=weights)
                .paint(resampler=resampler, interlacing=3, compensate=True)
                .value ** 2
            )

        grads_custom = jax.jit(jax.grad(loss, argnums=(0, 1)))(positions, weights)

        namespace = getattr(resamplers, resampler)
        original_paint = namespace.paint
        namespace.paint = partial(resamplers._paint_impl, order=order)
        try:
            grads_reference = jax.jit(jax.grad(loss, argnums=(0, 1)))(
                positions, weights
            )
        finally:
            namespace.paint = original_paint

        for custom, reference in zip(grads_custom, grads_reference):
            _assert_close(custom, reference)
            assert np.isfinite(np.asarray(custom)).all()
    print('interlaced/compensated gradient: ok')


if __name__ == '__main__':
    test_forward_parity()
    test_paint_gradients()
    test_read_gradients()
    test_scalar_weights()
    test_jit_gradients()
    test_particlefield_paint_gradient()
    test_realmeshfield_read_gradient()
    test_interlaced_compensated_gradient()
    print('all resampler adjoint tests passed')
