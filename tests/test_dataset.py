import numpy as np
import trimesh

from neural_mesh_simplification.data.dataset import (
    preprocess_mesh,
    load_mesh,
)


def test_load_mesh(tmp_path):
    # Create a temporary mesh file
    mesh = trimesh.creation.box()
    file_path = tmp_path / "test_mesh.obj"
    mesh.export(file_path)

    loaded_mesh = load_mesh(str(file_path))
    assert isinstance(loaded_mesh, trimesh.Trimesh)
    assert np.allclose(loaded_mesh.vertices, mesh.vertices)
    assert np.array_equal(loaded_mesh.faces, mesh.faces)


def test_preprocess_mesh_centered(sample_mesh):
    processed_mesh = preprocess_mesh(sample_mesh)
    # Check that the mesh is centered
    assert np.allclose(
        processed_mesh.vertices.mean(axis=0), np.zeros(3)
    ), "Mesh is not centered"


def test_preprocess_mesh_scaled(sample_mesh):
    processed_mesh = preprocess_mesh(sample_mesh)

    max_dim = np.max(
        processed_mesh.vertices.max(axis=0) - processed_mesh.vertices.min(axis=0)
    )
    assert np.isclose(max_dim, 1.0), "Mesh is not scaled to unit cube"
