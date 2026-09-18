import logging

import numpy as np
import pytest

import muxp_performance
from muxp_performance import (
    NumbaUnavailableError,
    apply_triangle_elevations,
    resolve_backend,
    triangles_to_array,
    update_elevation_array,
    update_ramp_array,
)
from xplnedsf2 import XPLNEpatch, XPLNEDSF, clearDSFpropertiesCache, getDSFproperties


def square_polygon():
    return np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=np.float64,
    )


def sample_triangles(count=1):
    base = np.array(
        [
            [[0.20, 0.20, 10.0], [0.80, 0.20, 20.0], [0.20, 0.80, 30.0]],
        ],
        dtype=np.float64,
    )
    return np.repeat(base, count, axis=0)


def write_synthetic_dsf(filename):
    dsf = XPLNEDSF("test_muxp_performance", None)
    dsf.Properties = {
        "sim/west": "0",
        "sim/east": "1",
        "sim/south": "0",
        "sim/north": "1",
        "sim/overlay": "0",
    }
    dsf.DefTerrains = {0: "lib/g10/terrain10/apt_tmp_grass.ter"}
    dsf.V = [[
        [0.20, 0.20, 10.0],
        [0.80, 0.20, 20.0],
        [0.20, 0.80, 30.0],
    ]]
    dsf.Scalings = [[
        [1.0, 0.0],
        [1.0, 0.0],
        [100.0, 0.0],
    ]]
    patch = XPLNEpatch(1, 0, -1, 0, 0)
    patch.trias2cmds([[[0, 0], [0, 1], [0, 2]]])
    dsf.Patches = [patch]
    # Keep the top-level atom wrappers so this is a valid minimal DSF, not
    # merely a file that the permissive reader happens to accept.
    dsf._Atoms_ = {
        "DAEH": [],
        "PORP": b"",
        "NFED": [],
        "TRET": b"",
        "TJBO": b"",
        "YLOP": b"",
        "WTEN": b"",
        "DOEG": [],
        "LOOP": [],
        "LACS": [],
        "23OP": [],
        "23CS": [],
        "SDMC": b"",
    }
    dsf.write(str(filename))


def update_synthetic_dsf(source, destination, backend):
    dsf = XPLNEDSF("test_muxp_performance", None)
    assert dsf.read(str(source)) == 0
    indexes = dsf.Patches[0].triangles()[0]
    triangles = [[[dsf.V[pool][vertex][:] for pool, vertex in indexes]]]
    coordinates = triangles_to_array(triangles[0])
    updated = update_elevation_array(
        coordinates,
        square_polygon(),
        42.0,
        backend=backend,
        workers=2,
    )
    apply_triangle_elevations(triangles[0], updated)
    for vertex_index, (pool, vertex) in enumerate(indexes):
        dsf.V[pool][vertex][2] = triangles[0][0][vertex_index][2]
    assert dsf.write(str(destination)) == 0


def test_triangle_array_is_float64_copy_and_preserves_input():
    source = [
        [[0, 0, 1], [1, 0, 2], [0, 1, 3]],
    ]
    result = triangles_to_array(source)
    assert result.shape == (1, 3, 3)
    assert result.dtype == np.float64
    result[0, 0, 2] = 99
    assert source[0][0][2] == 1


def test_elevation_inside_outside_and_boundary_cases():
    vertices = np.array(
        [
            [[0.5, 0.5, 1.0], [1.5, 0.5, 2.0], [0.0, 0.5, 3.0]],
        ],
        dtype=np.float64,
    )
    original = vertices.copy()
    updated = update_elevation_array(vertices, square_polygon(), 42.0, backend="python")
    assert np.array_equal(vertices, original)
    assert updated[0, 0, 2] == 42.0
    assert updated[0, 1, 2] == 2.0
    # The boundary behavior is deliberately identical to muxp_math.PointInPoly.
    assert updated[0, 2, 2] == 42.0


def test_ramp_elevation_and_outside_vertex():
    vertices = np.array(
        [
            [[0.0, 0.0, -1.0], [0.5, 0.0, -1.0], [1.5, 0.5, -1.0]],
        ],
        dtype=np.float64,
    )
    ramp = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 100.0], [0.0, 1.0, 200.0]],
        dtype=np.float64,
    )
    updated = update_ramp_array(vertices, square_polygon(), ramp, backend="python")
    assert updated[0, 0, 2] == 0.0
    assert updated[0, 1, 2] == 50.0
    assert updated[0, 2, 2] == -1.0
    assert np.array_equal(vertices[:, :, 2], np.full((1, 3), -1.0))


@pytest.mark.skipif(not muxp_performance.numba_available(), reason="optional Numba is not installed")
def test_python_and_numba_ramp_arrays_match():
    vertices = sample_triangles(4)
    ramp = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 100.0], [0.0, 1.0, 200.0]],
        dtype=np.float64,
    )
    python_result = update_ramp_array(vertices, square_polygon(), ramp, backend="python", workers=2)
    numba_result = update_ramp_array(vertices, square_polygon(), ramp, backend="numba", workers=2)
    assert np.array_equal(python_result, numba_result)


@pytest.mark.skipif(not muxp_performance.numba_available(), reason="optional Numba is not installed")
def test_numba_worker_limit_is_respected(monkeypatch):
    config = muxp_performance.numba.config
    original_limit = config.NUMBA_NUM_THREADS
    original_workers = muxp_performance.numba.get_num_threads()
    monkeypatch.setattr(config, "NUMBA_NUM_THREADS", 2)
    try:
        info = resolve_backend("numba", 1024, 4)
        assert info.workers == 2
        assert muxp_performance._set_numba_threads(4) == 2
        assert muxp_performance.numba.get_num_threads() == 2
    finally:
        monkeypatch.setattr(config, "NUMBA_NUM_THREADS", original_limit)
        muxp_performance.numba.set_num_threads(original_workers)


def test_backend_threshold_and_worker_validation():
    assert resolve_backend("python", 500, 1).selected == "python"
    assert resolve_backend("auto", 10, 1).selected == "python"
    with pytest.raises(ValueError):
        resolve_backend("invalid", 1, 1)
    with pytest.raises(ValueError):
        resolve_backend("python", 1, 0)


def test_explicit_numba_requires_importable_numba(monkeypatch):
    monkeypatch.setattr(muxp_performance, "numba", None)
    assert resolve_backend("auto", 1024, 1).selected == "python"
    with pytest.raises(NumbaUnavailableError):
        resolve_backend("numba", 1024, 1)


def test_synthetic_dsf_round_trip_and_property_cache(tmp_path, caplog):
    source = tmp_path / "source.dsf"
    output = tmp_path / "output.dsf"
    write_synthetic_dsf(source)
    clearDSFpropertiesCache()

    with caplog.at_level(logging.INFO):
        assert getDSFproperties(str(source))[0] == 0
        first_properties = getDSFproperties(str(source))[1]
    first_properties["sim/west"] = "changed outside cache"
    assert getDSFproperties(str(source))[1]["sim/west"] == "0"
    update_synthetic_dsf(source, output, "python")

    result = XPLNEDSF("test_muxp_performance", None)
    assert result.read(str(output)) == 0
    assert result.V[0][0][2] == pytest.approx(42.0, abs=0.01)


@pytest.mark.skipif(not muxp_performance.numba_available(), reason="optional Numba is not installed")
def test_python_and_numba_synthetic_dsf_bytes_match(tmp_path):
    source = tmp_path / "source.dsf"
    python_output = tmp_path / "python.dsf"
    numba_output = tmp_path / "numba.dsf"
    write_synthetic_dsf(source)
    update_synthetic_dsf(source, python_output, "python")
    update_synthetic_dsf(source, numba_output, "numba")
    assert python_output.read_bytes() == numba_output.read_bytes()
