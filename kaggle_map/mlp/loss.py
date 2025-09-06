"""Loss functions for misconception prediction."""

import torch
from torch import nn


class ListMLELoss(nn.Module):
    """ListMLE loss for learning to rank, optimized for MAP@3."""

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Compute ListMLE loss for ranking.

        Args:
            scores: [batch_size, n_classes] - predicted scores for each class
            labels: [batch_size] - ground truth class indices
            k: Top-k positions to optimize for (default 3 for MAP@3)

        Returns:
            Scalar loss value
        """
        batch_size, n_classes = scores.shape

        # Create one-hot encoding of labels
        labels_one_hot = torch.zeros_like(scores)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)

        # Sort scores in descending order
        sorted_scores, indices = torch.sort(scores, dim=1, descending=True)

        # Reorder labels according to sorted scores
        sorted_labels = labels_one_hot.gather(1, indices)

        # Focus on top-k positions
        top_k = min(k, n_classes)
        top_k_scores = sorted_scores[:, :top_k]
        top_k_labels = sorted_labels[:, :top_k]

        # Compute ListMLE loss
        exp_scores = torch.exp(top_k_scores)
        cumsum_exp_scores = torch.cumsum(exp_scores, dim=1)

        # Avoid log(0)
        epsilon = 1e-10
        log_probs = top_k_scores - torch.log(cumsum_exp_scores + epsilon)

        # Weight by ground truth labels
        loss = -torch.sum(top_k_labels * log_probs, dim=1)

        return loss.mean()
