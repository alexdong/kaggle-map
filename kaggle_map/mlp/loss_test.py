"""Tests for ListMLELoss."""

import torch

from kaggle_map.mlp.loss import ListMLELoss


def test_listmle_loss_initialization() -> None:
    """Test that ListMLELoss can be initialized."""
    loss_fn = ListMLELoss()
    assert isinstance(loss_fn, torch.nn.Module)


def test_listmle_loss_basic_computation() -> None:
    """Test basic loss computation with simple inputs."""
    loss_fn = ListMLELoss()

    # Create simple test data
    batch_size = 2
    n_classes = 5
    scores = torch.randn(batch_size, n_classes)
    labels = torch.tensor([1, 3])

    loss = loss_fn(scores, labels)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Scalar output
    assert loss.item() >= 0  # Loss should be non-negative


def test_listmle_loss_with_different_k() -> None:
    """Test loss computation with different k values."""
    loss_fn = ListMLELoss()

    scores = torch.randn(3, 10)
    labels = torch.tensor([0, 5, 9])

    # Test with default k=3
    loss_k3 = loss_fn(scores, labels, k=3)
    assert isinstance(loss_k3, torch.Tensor)

    # Test with k=5
    loss_k5 = loss_fn(scores, labels, k=5)
    assert isinstance(loss_k5, torch.Tensor)

    # Test with k larger than n_classes
    loss_k15 = loss_fn(scores, labels, k=15)
    assert isinstance(loss_k15, torch.Tensor)


def test_listmle_loss_perfect_ranking() -> None:
    """Test loss when predictions perfectly match labels."""
    loss_fn = ListMLELoss()

    # Create scores where correct class has highest score
    batch_size = 3
    n_classes = 4
    scores = torch.ones(batch_size, n_classes) * -10
    labels = torch.tensor([0, 2, 3])

    # Set correct class scores to be highest
    for i, label in enumerate(labels):
        scores[i, label] = 10.0

    loss = loss_fn(scores, labels)

    # Loss should be relatively small for perfect predictions
    assert loss.item() < 0.1


def test_listmle_loss_gradient_flow() -> None:
    """Test that gradients flow through the loss."""
    loss_fn = ListMLELoss()

    scores = torch.randn(2, 5, requires_grad=True)
    labels = torch.tensor([1, 3])

    loss = loss_fn(scores, labels)
    loss.backward()

    assert scores.grad is not None
    assert scores.grad.shape == scores.shape
    assert not torch.isnan(scores.grad).any()


def test_listmle_loss_batch_independence() -> None:
    """Test that batch samples are processed independently."""
    loss_fn = ListMLELoss()

    # Process full batch
    scores = torch.randn(2, 5)
    labels = torch.tensor([1, 3])
    batch_loss = loss_fn(scores, labels)

    # Process samples individually
    loss1 = loss_fn(scores[0:1], labels[0:1])
    loss2 = loss_fn(scores[1:2], labels[1:2])

    # Mean of individual losses should equal batch loss
    individual_mean = (loss1 + loss2) / 2
    assert torch.allclose(batch_loss, individual_mean, rtol=1e-5)


def test_listmle_loss_edge_cases() -> None:
    """Test edge cases."""
    loss_fn = ListMLELoss()

    # Single sample
    scores = torch.randn(1, 5)
    labels = torch.tensor([2])
    loss = loss_fn(scores, labels)
    assert isinstance(loss, torch.Tensor)

    # Binary classification
    scores = torch.randn(3, 2)
    labels = torch.tensor([0, 1, 0])
    loss = loss_fn(scores, labels)
    assert isinstance(loss, torch.Tensor)

    # k=1 (only top-1)
    scores = torch.randn(2, 5)
    labels = torch.tensor([1, 3])
    loss = loss_fn(scores, labels, k=1)
    assert isinstance(loss, torch.Tensor)
