import dgl
import torch
import torch.nn as nn
from dgl import DGLGraph

from .edge_predictor import EdgePredictorDGL
from .face_classifier import FaceClassifierDGL
from .point_sampler import PointSamplerDGL


class NeuralMeshSimplification(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        edge_hidden_dim: int,
        num_layers: int,
        k: int,
        edge_k: int,
        target_ratio: float,
        device=torch.device("cpu"),
    ):
        super(NeuralMeshSimplification, self).__init__()
        self.device = device
        self.k = k
        self.target_ratio = target_ratio

        self.point_sampler = PointSamplerDGL(input_dim, hidden_dim, num_layers).to(device)
        self.edge_predictor = EdgePredictorDGL(input_dim, edge_hidden_dim, edge_k).to(device)
        self.face_classifier = FaceClassifierDGL(input_dim, hidden_dim, num_layers, k).to(device)

    def forward(
        self,
        g: dgl.DGLGraph,
    ) -> tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        """
        Forward pass for NeuralMeshSimplification.

        Args:
            g (dlg.DGLGraph): Input graph containing node features `x` and optionally positions `pos`.

        Returns:
            dlg.DGLGraph: The graph containing the simplified mesh
            torch.Tensor: The simplified faces
            torch.Tensor: The face probabilities from the Face Classifier
        """
        x = g.ndata['x']
        pos = g.ndata['pos'] if 'pos' in g.ndata else x

        # Step 1: Sample points using the PointSamplerDGL
        sampled_indices, sampled_probs = self.sample_points(g)

        # Extract sampled features and positions
        sampled_x = x[sampled_indices]
        sampled_pos = pos[sampled_indices]

        # Create a new subgraph with sampled nodes
        sampled_g = dgl.node_subgraph(g, sampled_indices)

        # Step 2: Predict edges using EdgePredictorDGL
        edge_index_pred, edge_probs = self.edge_predictor(sampled_g)

        # Filter edges to keep only those connecting existing nodes
        # valid_edges = (edge_index_pred[0] < sampled_indices.shape[0]) & (edge_index_pred[1] < sampled_indices.shape[0])
        # edge_index_pred = edge_index_pred[:, valid_edges]
        # edge_probs = edge_probs[valid_edges]

        # Step 3: Generate candidate triangles
        candidate_triangles, triangle_probs = self.generate_candidate_triangles(sampled_g, edge_probs)

        # Step 4: Classify faces using FaceClassifierDGL
        if candidate_triangles.shape[0] > 0:
            # Create features and positions for triangles
            triangle_features = sampled_x[candidate_triangles].mean(dim=1)
            triangle_centers = sampled_pos[candidate_triangles].mean(dim=1)

            # Create a new DGL graph for the triangles
            triangle_g = dgl.graph(([], []), num_nodes=candidate_triangles.shape[0])
            triangle_g.ndata['x'] = triangle_features
            triangle_g.ndata['pos'] = triangle_centers

            # Classify faces
            face_probs = self.face_classifier(triangle_g, triangle_centers)
        else:
            face_probs = torch.empty(0, device=self.device)

        # Step 5: Filter triangles based on face probabilities
        if candidate_triangles.shape[0] == 0:
            simplified_faces = torch.empty(
                (0, 3), dtype=torch.long, device=self.device
            )
        else:
            threshold = torch.quantile(
                face_probs, 1 - self.target_ratio
            )  # Use a dynamic threshold
            simplified_faces = candidate_triangles[face_probs > threshold]

        # Create a new DGLGraph for the simplified mesh
        simplified_g = dgl.graph(
            (edge_index_pred[0], edge_index_pred[1]),
            num_nodes=sampled_indices.shape[0]
        )
        # Ensure all sampled vertices are included
        all_nodes = torch.arange(sampled_indices.shape[0], device=self.device)
        simplified_g = dgl.add_self_loop(simplified_g)
        simplified_g = dgl.add_edges(simplified_g, all_nodes, all_nodes)

        simplified_g.ndata['pos'] = sampled_pos
        simplified_g.ndata['x'] = sampled_x
        simplified_g.ndata['sample_prob'] = sampled_probs

        return simplified_g, simplified_faces, face_probs

    def sample_points(self, g: DGLGraph) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points using the PointSamplerDGL module.

        Args:
            g (DGLGraph): Input graph.

        Returns:
            tuple: Sampled indices and their probabilities.
        """
        num_nodes = g.num_nodes()

        # Determine the target number of nodes to sample
        target_nodes = min(max(int(self.target_ratio * num_nodes), 1), num_nodes)

        # Get sampling probabilities from PointSamplerDGL
        sampled_probs = self.point_sampler(g)

        # Select top-k nodes based on probabilities
        sampled_indices = torch.topk(sampled_probs, k=target_nodes).indices

        return sampled_indices, sampled_probs[sampled_indices]

    def generate_candidate_triangles(
        self,
        g: DGLGraph,
        edge_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate triangles from edges.

        Args:
            g (DGLGraph): Input graph with predicted edges.
            edge_probs (torch.Tensor): Probabilities of edges in the graph.

        Returns:
            tuple: Candidate triangles and their probabilities.
        """
        g = g.to(self.device)
        num_nodes = g.num_nodes()
        k = min(self.k, num_nodes - 1)

        # Create a feature matrix from edge probabilities
        src, dst = g.edges()
        features = torch.zeros((num_nodes, num_nodes), device=self.device)
        features[src, dst] = edge_probs
        features[dst, src] = edge_probs  # Assuming undirected graph

        del src
        del dst

        # Find k-nearest neighbors using dgl.knn_graph
        knn_g = dgl.knn_graph(features, k)
        knn_indices = knn_g.edges()[1].reshape(-1, k)

        del features
        del knn_g

        # Generate candidate triangles
        triangles = []
        triangle_probs = []

        for i in range(num_nodes):
            neighbors = knn_indices[i]
            for j in range(k):
                for l in range(j + 1, k):
                    n1, n2 = neighbors[j].item(), neighbors[l].item()
                    if g.has_edges_between(n1, n2):
                        triangles.append(torch.tensor([i, n1, n2], device=self.device))
                        e1 = g.edge_ids(i, n1)
                        e2 = g.edge_ids(i, n2)
                        e3 = g.edge_ids(n1, n2)
                        prob = (edge_probs[e1] * edge_probs[e2] * edge_probs[e3]) ** (1 / 3)
                        triangle_probs.append(prob)

        if triangles:
            triangles = torch.stack(triangles)
            triangle_probs = torch.stack(triangle_probs)
        else:
            triangles = torch.empty((0, 3), dtype=torch.long, device=self.device)
            triangle_probs = torch.empty(0, device=self.device)

        return triangles, triangle_probs
