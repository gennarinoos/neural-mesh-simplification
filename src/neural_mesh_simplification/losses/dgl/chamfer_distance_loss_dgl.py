import dgl
import torch
import torch.nn as nn


class ProbabilisticChamferDistanceLoss(nn.Module):
    def __init__(self):
        super(ProbabilisticChamferDistanceLoss, self).__init__()

    def forward(self, original_g: dgl.DGLGraph, simplified_g: dgl.DGLGraph):
        P = original_g.ndata['pos']
        Ps = simplified_g.ndata['pos']
        probabilities = simplified_g.ndata['sample_prob']

        if P.size(0) == 0 or Ps.size(0) == 0:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        # Compute distances from Ps to P
        dist_s_to_o = self.compute_minimum_distances(Ps, P)

        # Compute distances from P to Ps
        dist_o_to_s, min_indices = self.compute_minimum_distances(P, Ps, return_indices=True)

        # Weight distances by probabilities
        weighted_dist_s_to_o = dist_s_to_o * probabilities
        weighted_dist_o_to_s = dist_o_to_s * probabilities[min_indices]

        # Sum up the weighted distances
        loss = weighted_dist_s_to_o.sum() + weighted_dist_o_to_s.sum()

        return loss

    def compute_minimum_distances(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        return_indices: bool = False
    ):
        """
        Compute the minimum distances from each point in source to target.

        Args:
            source (torch.Tensor): Source point cloud, shape (N, 3)
            target (torch.Tensor): Target point cloud, shape (M, 3)
            return_indices (bool): If True, also return indices of minimum distances

        Returns:
            torch.Tensor: Minimum distances, shape (N,)
            torch.Tensor: Indices of minimum distances (if return_indices is True)
        """
        # Compute pairwise distances
        distances = torch.cdist(source, target)
        min_distances, min_indices = distances.min(dim=1)

        if return_indices:
            return min_distances, min_indices
        else:
            return min_distances
