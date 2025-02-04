import dgl
import torch
import torch.nn as nn
from dgl import DGLGraph

from .edge_predictor import EdgePredictorDGL
from .face_classifier import FaceClassifierDGL
from .point_sampler import PointSamplerDGL
from ..data.dataset import store_faces


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
        g: dgl.DGLGraph
    ) -> tuple[dgl.DGLGraph, torch.Tensor]:
        """
        Forward pass for NeuralMeshSimplification.

        Args:
            g (DGLGraph): Input graph containing node features `x` and optionally positions `pos`.

        Returns:
            DGLGraph: The graph containing the simplified mesh
            Tensor: The face probabilities from the Face Classifier
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
        store_faces(simplified_g, simplified_faces)

        return simplified_g, face_probs

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

    # TODO: Verify this optimized version

    def generate_candidate_triangles(
        self,
        sampled_g: DGLGraph,
        edge_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        print(f"sampled_g shape: {sampled_g.num_nodes()}, {sampled_g.num_edges()}")
        print(f"edge_probs shape: {edge_probs.shape}")

        # Get the adjacency matrix
        adj = sampled_g.adj().to_dense()
        print(f"adj shape: {adj.shape}")

        # Get k-nearest neighbors for each vertex
        k = min(self.k, adj.shape[0] - 1)
        _, knn_indices = torch.topk(adj, k=k, dim=1)

        print(f"knn_indices shape: {knn_indices.shape}")

        # Generate all possible triangles
        triangles, triangle_probs = self.generate_triangles_vectorized(knn_indices, edge_probs)

        print(f"triangles shape: {triangles.shape}")
        print(f"triangle_probs shape: {triangle_probs.shape}")

        return triangles, triangle_probs

    def generate_triangles_vectorized(self, knn_indices, edge_probs):
        num_vertices = knn_indices.shape[0]
        device = knn_indices.device

        # Generate all combinations of neighbor pairs
        neighbor_combinations = torch.combinations(torch.arange(self.k, device=device), r=2)

        # Create triangles: (center_vertex, neighbor1, neighbor2)
        center_vertices = (torch.arange(num_vertices, device=device)
                           .unsqueeze(1)
                           .unsqueeze(2)
                           .expand(-1, neighbor_combinations.shape[0], 1))
        neighbors = knn_indices[:, neighbor_combinations]
        triangles = torch.cat([center_vertices, neighbors], dim=2).reshape(-1, 3)

        # TODO: Calculate triangles probabilities

        # # Remove degenerate triangles
        # mask = (
        #     (triangles[:, 0] != triangles[:, 1])
        #     & (triangles[:, 0] != triangles[:, 2])
        #     & (triangles[:, 1] != triangles[:, 2])
        # )
        # triangles = triangles[mask]

        # Calculate triangle probabilities
        # edge_probs_expanded = edge_probs.unsqueeze(0).expand(num_vertices, -1, -1)
        # probs = torch.stack([
        #     edge_probs_expanded[triangles[:, 0], triangles[:, 1], triangles[:, 2]],
        #     edge_probs_expanded[triangles[:, 1], triangles[:, 0], triangles[:, 2]],
        #     edge_probs_expanded[triangles[:, 2], triangles[:, 0], triangles[:, 1]]
        # ], dim=1)
        # triangle_probs = probs.prod(dim=1).pow(1 / 3)
        triangle_probs = torch.zeros(triangles.shape[0])

        return triangles, triangle_probs

    # def generate_candidate_triangles(self, sampled_g, edge_probs):
    #     # Get the adjacency matrix
    #     adj = sampled_g.adj().to_dense()
    #
    #     # Get k-nearest neighbors for each vertex
    #     k = min(self.k, adj.shape[0] - 1)
    #     _, knn_indices = torch.topk(adj, k=k, dim=1)
    #
    #     # Initialize lists to store triangle indices and probabilities
    #     triangle_indices = []
    #     triangle_probs = []
    #
    #     # Iterate over all vertices
    #     for i in range(adj.shape[0]):
    #         neighbors = knn_indices[i]
    #         # Generate all possible triangles with the current vertex and its neighbors
    #         for j in range(k):
    #             for l in range(j + 1, k):
    #                 n1, n2 = neighbors[j], neighbors[l]
    #                 # Ensure we don't create degenerate triangles
    #                 if i != n1 and i != n2 and n1 != n2:
    #                     triangle = torch.tensor([i, n1, n2], device=self.device)
    #                     triangle_indices.append(triangle)
    #
    #                     # Calculate triangle probability based on edge probabilities
    #                     prob = (edge_probs[i, n1] * edge_probs[i, n2] * edge_probs[n1, n2]) ** (1 / 3)
    #                     triangle_probs.append(prob)
    #
    #     # Convert lists to tensors
    #     triangle_indices = torch.stack(triangle_indices)
    #     triangle_probs = torch.tensor(triangle_probs, device=self.device)
    #
    #     return triangle_indices, triangle_probs

    # def generate_candidate_triangles(
    #     self,
    #     g: DGLGraph,
    #     edge_probs: torch.Tensor
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Generate candidate triangles from edges.
    #
    #     Args:
    #         g (DGLGraph): Input graph with predicted edges.
    #         edge_probs (torch.Tensor): Probabilities of edges in the graph.
    #
    #     Returns:
    #         tuple: Candidate triangles and their probabilities.
    #     """
    #     # Ensure the graph is on the correct device
    #     g = g.to(self.device)
    #
    #     # Get the number of nodes
    #     num_nodes = g.num_nodes()
    #
    #     # Create an adjacency matrix from the graph
    #     adj_matrix = g.adj().to_dense()
    #
    #     # Set edge probabilities
    #     adj_matrix[g.edges()] = edge_probs
    #
    #     # Adjust k based on the number of nodes
    #     k = min(self.k, num_nodes - 1)
    #
    #     # Find k-nearest neighbors for each node
    #     _, knn_indices = torch.topk(adj_matrix, k=k, dim=1)
    #
    #     # Generate candidate triangles
    #     triangles = []
    #     triangle_probs = []
    #
    #     for i in range(num_nodes):
    #         neighbors = knn_indices[i]
    #         for j in range(k):
    #             for l in range(j + 1, k):
    #                 n1, n2 = neighbors[j], neighbors[l]
    #                 if adj_matrix[n1, n2] > 0:  # Check if the third edge exists
    #                     triangle = torch.tensor([i, n1, n2], device=self.device)
    #                     triangles.append(triangle)
    #
    #                     # Calculate triangle probability
    #                     prob = (
    #                                adj_matrix[i, n1] * adj_matrix[i, n2] * adj_matrix[n1, n2]
    #                            ) ** (1 / 3)
    #                     triangle_probs.append(prob)
    #
    #     if triangles:
    #         triangles = torch.stack(triangles)
    #         triangle_probs = torch.tensor(triangle_probs, device=self.device)
    #     else:
    #         triangles = torch.empty((0, 3), dtype=torch.long, device=self.device)
    #         triangle_probs = torch.empty(0, device=self.device)
    #
    #     return triangles, triangle_probs
