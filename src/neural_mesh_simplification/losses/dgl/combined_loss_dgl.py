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

    def forward(
        self,
        original_graph: dgl.DGLGraph,
        original_faces: torch.Tensor,
        sampled_graph: dgl.DGLGraph,
        sampled_faces: torch.Tensor,
        face_probs: torch.Tensor
    ):
        orig_vertices = original_graph.ndata['x'].to(self.device)
        sampled_vertices = sampled_graph.ndata['x'].to(self.device)
        sampled_probs = sampled_graph.ndata['sampled_prob'].to(self.device)

        del original_graph

        chamfer_loss = self.prob_chamfer_loss(orig_vertices, sampled_vertices, sampled_probs)
        surface_loss = self.prob_surface_loss(orig_vertices, original_faces, sampled_vertices, sampled_faces, face_probs)
        collision_loss = self.collision_loss(sampled_vertices, sampled_faces, face_probs)
        edge_crossing_loss = self.edge_crossing_loss(sampled_vertices, sampled_faces, face_probs)
        overlapping_triangles_loss = self.overlapping_triangles_loss(sampled_vertices, sampled_faces)

        total_loss = (
            chamfer_loss
            + surface_loss
            + self.lambda_c * collision_loss
            + self.lambda_e * edge_crossing_loss
            + self.lambda_o * overlapping_triangles_loss
        )

        return total_loss
