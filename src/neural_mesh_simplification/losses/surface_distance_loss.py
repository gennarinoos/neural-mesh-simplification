import torch
import torch.nn as nn
from torch_cluster import knn


class ProbabilisticSurfaceDistanceLoss(nn.Module):
    def __init__(self, k: int = 3, num_samples: int = 100, epsilon: float = 1e-8):
        super().__init__()
        self.k = k
        self.num_samples = num_samples
        self.epsilon = epsilon
        print(
            f"Initialized ProbabilisticSurfaceDistanceLoss with k={self.k}, num_samples={self.num_samples}"
        )

    def forward(
            self,
            original_vertices: torch.Tensor,
            original_faces: torch.Tensor,
            simplified_vertices: torch.Tensor,
            simplified_faces: torch.Tensor,
            face_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        if original_vertices.shape[0] == 0 or simplified_vertices.shape[0] == 0:
            return torch.tensor(0.0, device=original_vertices.device)

        forward_term = self.compute_forward_term(
            original_vertices,
            original_faces,
            simplified_vertices,
            simplified_faces,
            face_probabilities,
        )

        reverse_term = self.compute_reverse_term(
            original_vertices,
            original_faces,
            simplified_vertices,
            simplified_faces,
            face_probabilities,
        )

        total_loss = forward_term + reverse_term
        return total_loss

    def compute_forward_term(
            self,
            original_vertices: torch.Tensor,
            original_faces: torch.Tensor,
            simplified_vertices: torch.Tensor,
            simplified_faces: torch.Tensor,
            face_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        # If there are no faces, return zero loss
        if simplified_faces.shape[0] == 0:
            return torch.tensor(0.0, device=original_vertices.device)

        simplified_barycenters = self.compute_barycenters(
            simplified_vertices, simplified_faces
        )
        original_barycenters = self.compute_barycenters(
            original_vertices, original_faces
        )

        distances = self.compute_squared_distances(
            simplified_barycenters, original_barycenters
        )

        min_distances, _ = distances.min(dim=1)

        # Ensure face_probabilities matches the number of simplified faces
        if face_probabilities.shape[0] > simplified_faces.shape[0]:
            face_probabilities = face_probabilities[: simplified_faces.shape[0]]
        elif face_probabilities.shape[0] < simplified_faces.shape[0]:
            # Pad with zeros if we have fewer probabilities than faces
            padding = torch.zeros(
                simplified_faces.shape[0] - face_probabilities.shape[0],
                device=face_probabilities.device,
            )
            face_probabilities = torch.cat([face_probabilities, padding])

        # Weight distances by face probabilities
        weighted_distances = face_probabilities * min_distances

        total_loss = weighted_distances.sum()

        # Ensure face probabilities affect the loss even when distances are zero
        probability_penalty = 1e-4 * (1.0 - face_probabilities).sum()

        total_loss += probability_penalty

        return total_loss

    def compute_reverse_term(
            self,
            original_vertices: torch.Tensor,
            original_faces: torch.Tensor,
            simplified_vertices: torch.Tensor,
            simplified_faces: torch.Tensor,
            face_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        # If there are no faces, return zero loss
        if simplified_faces.shape[0] == 0:
            return torch.tensor(0.0, device=original_vertices.device)

        # If meshes are identical, reverse term should be zero
        if torch.equal(original_vertices, simplified_vertices) and torch.equal(
                original_faces, simplified_faces
        ):
            return torch.tensor(0.0, device=original_vertices.device)

        # Step 1: Sample points from the simplified mesh
        sampled_points = self.sample_points_from_triangles(
            simplified_vertices, simplified_faces, self.num_samples
        )

        # Step 2: Compute the minimum distance from each sampled point to the original mesh
        min_distances_to_original = self.compute_min_distances_to_original(
            sampled_points, original_vertices
        ).float()

        # Normalize the distances to prevent large values
        normalized_distances = min_distances_to_original / (
                min_distances_to_original.max() + self.epsilon
        )

        # Further scaling to reduce the impact
        scaled_distances = normalized_distances * 0.1

        # Ensure face_probabilities matches the number of simplified faces
        if face_probabilities.shape[0] > simplified_faces.shape[0]:
            face_probabilities = face_probabilities[: simplified_faces.shape[0]]
        elif face_probabilities.shape[0] < simplified_faces.shape[0]:
            # Pad with zeros if we have fewer probabilities than faces
            padding = torch.zeros(
                simplified_faces.shape[0] - face_probabilities.shape[0],
                device=face_probabilities.device,
            )
            face_probabilities = torch.cat([face_probabilities, padding])
        # Reshape face probabilities to match the sampled points
        face_probabilities_expanded = face_probabilities.repeat_interleave(
            self.num_samples
        )

        # Weight by face probabilities
        weighted_min_distances = face_probabilities_expanded * scaled_distances

        # Return the sum as the reverse term
        reverse_term = weighted_min_distances.sum()

        return reverse_term

    def compute_min_distances_to_original(
            self, sampled_points: torch.Tensor, original_vertices: torch.Tensor
    ) -> torch.Tensor:
        distances, _ = knn(original_vertices.float(), sampled_points.float(), k=1)
        return distances.view(-1).float()  # Convert to float

    def compute_barycenters(
            self, vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        return vertices[faces].mean(dim=1)

    def sample_points_from_triangles(
            self, vertices: torch.Tensor, faces: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        num_faces = faces.shape[0]
        face_vertices = vertices[faces]

        r1 = torch.sqrt(torch.rand(num_faces, num_samples, 1, device=vertices.device))
        r2 = torch.rand(num_faces, num_samples, 1, device=vertices.device)

        a = 1 - r1
        b = r1 * (1 - r2)
        c = r1 * r2

        samples = (
                a * face_vertices[:, None, 0]
                + b * face_vertices[:, None, 1]
                + c * face_vertices[:, None, 2]
        )

        return samples.view(-1, 3)

    def compute_squared_distances(
            self, barycenters1: torch.Tensor, barycenters2: torch.Tensor
    ) -> torch.Tensor:
        num_faces1 = barycenters1.size(0)
        num_faces2 = barycenters2.size(0)

        barycenters1_exp = barycenters1.unsqueeze(1).expand(num_faces1, num_faces2, 3)
        barycenters2_exp = barycenters2.unsqueeze(0).expand(num_faces1, num_faces2, 3)

        distances = torch.sum((barycenters1_exp - barycenters2_exp) ** 2, dim=2)

        return distances.float()  # Convert to float
