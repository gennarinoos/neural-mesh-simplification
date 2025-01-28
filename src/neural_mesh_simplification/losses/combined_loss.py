import torch.nn as nn

from . import (
    ProbabilisticChamferDistanceLoss,
    ProbabilisticSurfaceDistanceLoss,
    TriangleCollisionLoss,
    EdgeCrossingLoss,
    OverlappingTrianglesLoss,
)


class CombinedMeshSimplificationLoss(nn.Module):
    def __init__(
        self, lambda_c: float = 1.0, lambda_e: float = 1.0, lambda_o: float = 1.0
    ):
        super().__init__()
        self.prob_chamfer_loss = ProbabilisticChamferDistanceLoss()
        self.prob_surface_loss = ProbabilisticSurfaceDistanceLoss()
        self.collision_loss = TriangleCollisionLoss()
        self.edge_crossing_loss = EdgeCrossingLoss()
        self.overlapping_triangles_loss = OverlappingTrianglesLoss()
        self.lambda_c = lambda_c
        self.lambda_e = lambda_e
        self.lambda_o = lambda_o

    def forward(self, original_data, simplified_data):
        original_x = (
            original_data["pos"] if "pos" in original_data else original_data["x"]
        )
        original_face = original_data["face"]
        sampled_indices = simplified_data["sampled_indices"]
        sampled_vertices = simplified_data["sampled_vertices"]
        sampled_probs = simplified_data["sampled_probs"]

        chamfer_loss = self.prob_chamfer_loss(
            original_x, sampled_vertices, sampled_probs
        )
        surface_loss = self.prob_surface_loss(
            original_x,
            original_face,
            sampled_vertices,
            simplified_data["simplified_faces"],
            simplified_data["face_probs"],
        )
        collision_loss = self.collision_loss(
            sampled_vertices,
            simplified_data["simplified_faces"],
            simplified_data["face_probs"],
        )
        edge_crossing_loss = self.edge_crossing_loss(simplified_data)
        overlapping_triangles_loss = self.overlapping_triangles_loss(simplified_data)

        total_loss = (
            chamfer_loss
            + surface_loss
            + self.lambda_c * collision_loss
            + self.lambda_e * edge_crossing_loss
            + self.lambda_o * overlapping_triangles_loss
        )

        return total_loss
