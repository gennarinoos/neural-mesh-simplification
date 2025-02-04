import dgl
import torch
import torch.nn as nn

from ...data.dataset import reconstruct_faces


class EdgeCrossingLoss(nn.Module):
    def __init__(self, k: int = 20):
        super().__init__()
        self.k = k  # Number of nearest triangles to consider

    def forward(self, g: dgl.DGLGraph, face_probs: torch.Tensor) -> torch.Tensor:
        vertices = g.ndata['pos']

        faces = reconstruct_faces(g)

        if faces.shape[0] == 0:
            return torch.tensor(0.0, device=vertices.device)

        # Ensure face_probs matches the number of faces
        face_probs = face_probs[:faces.shape[0]]

        # Find k-nearest triangles for each triangle
        nearest_triangles = self.find_nearest_triangles(g)

        # Detect edge crossings between nearby triangles
        crossings = self.detect_edge_crossings(g, nearest_triangles)

        # Calculate loss
        loss = self.calculate_loss(crossings, face_probs)

        return loss

    def find_nearest_triangles(self, g: dgl.DGLGraph) -> torch.Tensor:
        faces = reconstruct_faces(g)
        centroids = g.ndata['pos'][faces].mean(dim=1)
        k = min(self.k, centroids.shape[0])
        g_knn = dgl.knn_graph(centroids, k)
        return g_knn.edges()[1].reshape(-1, k)

    def detect_edge_crossings(self, g: dgl.DGLGraph, nearest_triangles: torch.Tensor) -> torch.Tensor:
        vertices = g.ndata['pos']
        faces = reconstruct_faces(g)

        def edge_vectors(triangles):
            return vertices[triangles[:, [1, 2, 0]]] - vertices[triangles]

        edges = edge_vectors(faces)
        crossings = torch.zeros(faces.shape[0], device=vertices.device)

        for i in range(faces.shape[0]):
            neighbor_edges = edge_vectors(faces[nearest_triangles[i]])
            for j in range(3):
                edge = edges[i, j].unsqueeze(0).unsqueeze(0)
                cross_product = torch.cross(edge.expand(neighbor_edges.shape), neighbor_edges, dim=-1)
                t = torch.sum(cross_product * neighbor_edges, dim=-1) / torch.sum(cross_product * edge.expand(neighbor_edges.shape), dim=-1)
                u = torch.sum(cross_product * edges[i].unsqueeze(0), dim=-1) / torch.sum(cross_product * edge.expand(neighbor_edges.shape), dim=-1)
                mask = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
                crossings[i] += mask.sum()

        return crossings

    def calculate_loss(self, crossings: torch.Tensor, face_probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(face_probs * crossings, dtype=torch.float32)
