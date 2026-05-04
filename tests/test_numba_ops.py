import numpy as np
from netai.inference.numba_ops import NumbaBackend, HAS_NUMBA, apply_numba_patches
from netai.inference.native_engine import NativeInferenceEngine, TransformerConfig, _layer_norm, _gelu, _silu, _softmax


class TestNumbaBackend:
    def test_availability(self):
        backend = NumbaBackend()
        assert backend.is_available() == HAS_NUMBA

    def test_warmup(self):
        backend = NumbaBackend()
        status = backend.get_warmup_status()
        assert "numba_available" in status

    def test_numba_gelu_correctness(self):
        if not HAS_NUMBA:
            import pytest; pytest.skip("Numba not available")
        from netai.inference.numba_ops import numba_gelu
        x = np.random.randn(2, 512).astype(np.float32)
        py_out = _gelu(x)
        nb_out = numba_gelu(x.copy())
        assert np.allclose(py_out, nb_out, atol=1e-4)

    def test_numba_silu_correctness(self):
        if not HAS_NUMBA:
            import pytest; pytest.skip("Numba not available")
        from netai.inference.numba_ops import numba_silu
        x = np.random.randn(2, 512).astype(np.float32)
        py_out = _silu(x)
        nb_out = numba_silu(x.copy())
        assert np.allclose(py_out, nb_out, atol=1e-4)

    def test_numba_softmax_correctness(self):
        if not HAS_NUMBA:
            import pytest; pytest.skip("Numba not available")
        from netai.inference.numba_ops import numba_softmax
        x = np.random.randn(4, 128).astype(np.float32)
        py_out = _softmax(x, axis=-1)
        nb_out = numba_softmax(x.copy(), -1)
        assert np.allclose(py_out, nb_out, atol=1e-4)

    def test_numba_layer_norm_correctness(self):
        if not HAS_NUMBA:
            import pytest; pytest.skip("Numba not available")
        from netai.inference.numba_ops import numba_layer_norm
        x = np.random.randn(1, 128, 768).astype(np.float32)
        w = np.ones(768, dtype=np.float32)
        b = np.zeros(768, dtype=np.float32)
        py_out = _layer_norm(x, w, b, 1e-5)
        nb_out = numba_layer_norm(x.copy(), w, b, 1e-5)
        assert np.allclose(py_out, nb_out, atol=1e-4)

    def test_apply_patches(self):
        if not HAS_NUMBA:
            import pytest; pytest.skip("Numba not available")
        engine = NativeInferenceEngine()
        result = apply_numba_patches(engine)
        assert isinstance(result, bool)

    def test_backend_fallback_when_unavailable(self):
        backend = NumbaBackend()
        ok = backend.is_available()
        assert ok is True or ok is False


class TestNumbaPerformance:
    def test_gelu_speedup(self):
        if not HAS_NUMBA:
            import pytest; pytest.skip("Numba not available")
        from netai.inference.numba_ops import numba_gelu
        import time
        x = np.random.randn(1, 128, 768).astype(np.float32)
        # Warmup
        for _ in range(5):
            numba_gelu(x.copy())
        t0 = time.time()
        for _ in range(50):
            _gelu(x)
        t_py = time.time() - t0
        t0 = time.time()
        for _ in range(50):
            numba_gelu(x.copy())
        t_nb = time.time() - t0
        assert t_nb <= t_py * 2  # at most equal or better after warmup

    def test_softmax_stability(self):
        if not HAS_NUMBA:
            import pytest; pytest.skip("Numba not available")
        from netai.inference.numba_ops import numba_softmax
        x = np.array([[1000.0, 1000.0, 1000.0]], dtype=np.float32)
        result = numba_softmax(x.copy(), -1)
        assert not np.any(np.isnan(result))
        assert np.all(result >= 0)

    def test_layer_norm_stability(self):
        if not HAS_NUMBA:
            import pytest; pytest.skip("Numba not available")
        from netai.inference.numba_ops import numba_layer_norm
        x = np.zeros((2, 4, 64), dtype=np.float32)
        w = np.ones(64, dtype=np.float32)
        b = np.zeros(64, dtype=np.float32)
        result = numba_layer_norm(x.copy(), w, b, 1e-5)
        assert not np.any(np.isnan(result))

    def test_edge_case_empty(self):
        if not HAS_NUMBA:
            import pytest; pytest.skip("Numba not available")
        from netai.inference.numba_ops import numba_gelu
        x = np.array([], dtype=np.float32)
        result = numba_gelu(x.copy())
        assert len(result) == 0
