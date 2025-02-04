import dgl
import torch
import torch.nn as nn

from .chamfer_distance_loss_dgl import ProbabilisticChamferDistanceLoss
from .edge_crossing_loss_dgl import EdgeCrossingLoss
from .overlapping_triangles_loss_dgl import OverlappingTrianglesLoss
from .surface_distance_loss_dgl import ProbabilisticSurfaceDistanceLoss
from .triangle_collision_loss_dgl import TriangleCollisionLoss


class CombinedMeshSimplificationLoss(nn.Module):
    def __init__(
        self,
        lambda_c: float = 1.0,
        lambda_e: float = 1.0,
        lambda_o: float = 1.0,
        device=torch.device("cpu")
    ):
        super().__init__()
        self.device = device
        self.prob_chamfer_loss = ProbabilisticChamferDistanceLoss().to(self.device)
        self.prob_surface_loss = ProbabilisticSurfaceDistanceLoss().to(self.device)
        self.collision_loss = TriangleCollisionLoss().to(self.device)
        self.edge_crossing_loss = EdgeCrossingLoss().to(self.device)
        self.overlapping_triangles_loss = OverlappingTrianglesLoss().to(self.device)
        self.lambda_c = lambda_c
        self.lambda_e = lambda_e
        self.lambda_o = lambda_o

    def forward(self, original_g: dgl.DGLGraph, simplified_g: dgl.DGLGraph, face_probs: torch.Tensor):
        chamfer_loss = self.prob_chamfer_loss(original_g, simplified_g)
        surface_loss = self.prob_surface_loss(original_g, simplified_g)

        del original_g

        collision_loss = self.collision_loss(simplified_g, face_probs)
        edge_crossing_loss = self.edge_crossing_loss(simplified_g, face_probs)
        overlapping_triangles_loss = self.overlapping_triangles_loss(simplified_g)

        total_loss = (
            chamfer_loss
            + surface_loss
            + self.lambda_c * collision_loss
            + self.lambda_e * edge_crossing_loss
            + self.lambda_o * overlapping_triangles_loss
        )

        return total_loss
