import trimesh

def save_mesh(mesh1: trimesh.Trimesh, path: str):
    # Export the mesh to OBJ format
    mesh1.export(path)
