import dgl
import torch
import torch.nn as nn

from neural_mesh_simplification.data.dataset import reconstruct_faces


class ProbabilisticSurfaceDistanceLoss(nn.Module):
    def __init__(self, num_samples: int = 100):
        super(ProbabilisticSurfaceDistanceLoss, self).__init__()
        self.num_samples = num_samples

    def forward(self, original_g: dgl.DGLGraph, simplified_g: dgl.DGLGraph):
        original_vertices = original_g.ndata['pos']
        simplified_vertices = simplified_g.ndata['pos']
        probabilities = simplified_g.ndata['sample_prob']
        original_faces = reconstruct_faces(original_g)
        simplified_faces = reconstruct_faces(simplified_g)

        # Sample points on original mesh
        original_samples = self.sample_points_on_mesh(original_vertices, original_faces)

        # Sample points on simplified mesh
        simplified_samples = self.sample_points_on_mesh(simplified_vertices, simplified_faces)

        # Compute distances
        dist_o_to_s = self.compute_minimum_distances(original_samples, simplified_samples)
        dist_s_to_o = self.compute_minimum_distances(simplified_samples, original_samples)

        # Weight distances by probabilities
        weighted_dist_s_to_o = dist_s_to_o * probabilities.unsqueeze(1)
        weighted_dist_o_to_s = dist_o_to_s

        # Compute loss
        loss = weighted_dist_s_to_o.mean() + weighted_dist_o_to_s.mean()

        return loss

    def sample_points_on_mesh(self, vertices, faces):
        num_faces = faces.shape[0]
        face_areas = self.compute_face_areas(vertices, faces)
        face_probs = face_areas / face_areas.sum()

        selected_faces = torch.multinomial(face_probs, self.num_samples, replacement=True)
        u = torch.rand(self.num_samples, device=vertices.device)
        v = torch.rand(self.num_samples, device=vertices.device)
        w = 1 - u - v
        mask = w < 0
        u[mask] += w[mask]
        v[mask] = 1 - u[mask]
        w[mask] = 0

        v0, v1, v2 = vertices[faces[selected_faces]].unbind(1)
        sampled_points = u.unsqueeze(1) * v0 + v.unsqueeze(1) * v1 + w.unsqueeze(1) * v2

        return sampled_points

    def compute_face_areas(self, vertices, faces):
        v0, v1, v2 = vertices[faces].unbind(1)
        return torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1) * 0.5

    def compute_minimum_distances(self, source, target):
        distances = torch.cdist(source, target)
        min_distances, _ = distances.min(dim=1)
        return min_distances
