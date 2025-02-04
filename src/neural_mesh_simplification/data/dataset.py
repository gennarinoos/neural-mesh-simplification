import os
from typing import Optional

import dgl
import numpy as np
import torch
import trimesh
from dgl.data import DGLDataset
from trimesh import Geometry, Trimesh


class MeshSimplificationDataset(DGLDataset):
    def __init__(self, data_dir, preprocess: bool = False, transform: Optional[callable] = None):
        super().__init__(name='mesh_simplification')
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.transform = transform
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        return [
            f
            for f in os.listdir(self.data_dir)
            if f.endswith(".ply") or f.endswith(".obj") or f.endswith(".stl")
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> dgl.DGLGraph:
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        mesh = load_mesh(file_path)

        if self.preprocess:
            mesh = preprocess_mesh(mesh)

        if self.transform:
            mesh = self.transform(mesh)

        graph = mesh_to_dgl(mesh)
        return graph


def load_mesh(file_path: str) -> Geometry | list[Geometry] | None:
    """Load a mesh from file."""
    try:
        mesh = trimesh.load(file_path)
        return mesh
    except Exception as e:
        print(f"Error loading mesh {file_path}: {e}")
        return None


def preprocess_mesh(mesh: trimesh.Trimesh) -> Trimesh | None:
    """Preprocess a mesh (e.g., normalize, center)."""
    if mesh is None:
        return None

    # Center the mesh
    mesh.vertices -= mesh.vertices.mean(axis=0)

    # Scale to unit cube
    max_dim = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    mesh.vertices /= max_dim

    return mesh


def augment_mesh(mesh: trimesh.Trimesh) -> Trimesh | None:
    """Apply data augmentation to a mesh."""
    if mesh is None:
        return None

    # Example: Random rotation
    rotation = trimesh.transformations.random_rotation_matrix()
    mesh.apply_transform(rotation)

    return mesh


def mesh_to_dgl(mesh) -> dgl.DGLGraph | None:
    if mesh is None:
        return None

    # Convert vertices to tensor
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    num_nodes = vertices.shape[0]

    # Convert unique edges
    edges_np = np.array(list(mesh.edges_unique))
    edges = torch.tensor(edges_np, dtype=torch.int64).t()

    # Create DGL graph
    g = dgl.graph((edges[0], edges[1]), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)

    # Add node features
    g.ndata['x'] = vertices
    g.ndata['pos'] = vertices

    # Store face information as node data
    if hasattr(mesh, 'faces'):
        faces = torch.tensor(mesh.faces, dtype=torch.int64)
        store_faces(g, faces)

    # Verify node count matches
    assert g.number_of_nodes() == vertices.shape[0], "Mismatch between nodes and features"

    return g


def store_faces(g: dgl.DGLGraph, faces: torch.Tensor):
    # Create a mapping from node indices to face indices
    node_to_face = torch.full((g.num_nodes(),), -1, dtype=torch.int64)  # Default: -1 (no face)
    for face_id, (v1, v2, v3) in enumerate(faces):
        node_to_face[v1] = face_id
        node_to_face[v2] = face_id
        node_to_face[v3] = face_id

    g.ndata['face_idx'] = node_to_face  # Store face indices per node


def reconstruct_faces(g: dgl.DGLGraph) -> torch.Tensor:
    if 'face_idx' not in g.ndata:
        raise IOError("No face data set on graph under key 'face_idx'")

    face_idx = g.ndata['face_idx']  # Retrieve stored face indices
    valid_mask = face_idx >= 0  # Ignore nodes with -1 (no face)

    # Filter only valid nodes
    valid_node_ids = torch.arange(g.num_nodes())[valid_mask]
    valid_face_ids = face_idx[valid_mask]  # Get valid face indices

    # Dictionary: {face_id: [list of node indices]}
    face_dict = {}
    for node_id, f_id in zip(valid_node_ids.tolist(), valid_face_ids.tolist()):
        if f_id not in face_dict:
            face_dict[f_id] = []
        face_dict[f_id].append(node_id)

    # Convert to tensor (only faces with exactly 3 nodes)
    faces = [nodes for nodes in face_dict.values() if len(nodes) == 3]
    faces_tensor = torch.tensor(faces, dtype=torch.int64) if faces else None

    return faces_tensor
