"""
Smoke test to verify basic pctree functionality and imports.
This test ensures the package is properly installed and core components work.
"""

import numpy as np
import pytest
from pathlib import Path


def test_imports():
    """Test that all main modules can be imported."""
    from pctree import core, training, branches, io
    from pctree.training import PCTreeTrainer, PCTreeTrainerOptions
    from pctree.branches import EfficientEncodingRouter
    from pctree.core import PCTree, PCTreeCoefficients

    assert core is not None
    assert training is not None
    assert branches is not None
    assert io is not None


def test_mnist_workflow():
    """Test PCT workflow using real MNIST data from the demos."""
    from pctree.training import PCTreeTrainer, PCTreeTrainerOptions
    from pctree.branches import EfficientEncodingRouter
    from pctree.core import PCTreeCoefficients

    # Load MNIST data (same as demo_start_here.ipynb)
    data_path = Path(__file__).parent.parent / "data" / "mnist_train.npy"

    # Skip test if data file doesn't exist
    if not data_path.exists():
        pytest.skip(f"MNIST data not found at {data_path}")

    mnist_X = np.load(str(data_path)).astype(np.float32)
    X_mean = mnist_X.mean(axis=0)
    mnist_X -= X_mean

    # Train a small PCT (using same params as notebook)
    trainer = PCTreeTrainer(PCTreeTrainerOptions(
        max_nodes=50,
        max_children=5,
        max_width=10,
        sifting_effect_size=0.6
    ))

    # Fit the tree
    trainer.fit_partition(mnist_X, verbose=0)
    tree = trainer.fit_em(mnist_X, X_topdown=mnist_X, verbose=0)

    # Basic assertions on tree structure
    assert tree is not None
    assert tree.V is not None
    assert tree.E is not None
    assert tree.s is not None
    assert len(tree.V) > 0
    assert len(tree.V) <= 50  # Should respect max_nodes

    # Test branch assignment
    router = EfficientEncodingRouter(tree)
    branch_assignments = router.predict(mnist_X)

    assert branch_assignments is not None
    assert len(branch_assignments) == len(mnist_X)
    assert branch_assignments.min() >= 0
    assert branch_assignments.max() < len(tree.branches())

    # Test encoding and reconstruction
    coeffs = PCTreeCoefficients(tree, mnist_X, branch_assignments)
    X_reconstructed = coeffs.reconstruct()

    assert X_reconstructed.shape == mnist_X.shape

    # Verify reconstruction captures reasonable variance
    variance_captured = 1 - (((mnist_X - X_reconstructed)**2).sum() / (mnist_X**2).sum())
    assert variance_captured > 0.5, f"Variance captured too low: {variance_captured:.2f}"


def test_save_load_roundtrip(tmp_path):
    """Test that PCTree can be saved and loaded using MNIST data."""
    from pctree.training import PCTreeTrainer, PCTreeTrainerOptions
    from pctree.branches import EfficientEncodingRouter
    from pctree.core import PCTreeCoefficients

    # Load MNIST data
    data_path = Path(__file__).parent.parent / "data" / "mnist_train.npy"

    # Skip test if data file doesn't exist
    if not data_path.exists():
        pytest.skip(f"MNIST data not found at {data_path}")

    mnist_X = np.load(str(data_path)).astype(np.float32)
    X_mean = mnist_X.mean(axis=0)
    mnist_X -= X_mean

    # Train a small tree
    trainer = PCTreeTrainer(PCTreeTrainerOptions(max_nodes=20, max_width=5))
    trainer.fit_partition(mnist_X, verbose=0)
    tree = trainer.fit_em(mnist_X, X_topdown=mnist_X, verbose=0)

    router = EfficientEncodingRouter(tree)
    branch_assignments = router.predict(mnist_X)
    coeffs = PCTreeCoefficients(tree, mnist_X, branch_assignments)

    # Save to temporary file
    save_path = tmp_path / "test_tree.npz"
    coeffs.save(str(save_path))

    # Verify file was created
    assert save_path.exists()

    # Load and verify basic properties match
    loaded_data = np.load(str(save_path))
    assert 'tree_V' in loaded_data
    assert 'tree_E' in loaded_data
    assert 'tree_s' in loaded_data
    assert 'branch-assignments' in loaded_data

    # Verify shapes match
    assert loaded_data['tree_V'].shape == tree.V.shape
    assert loaded_data['tree_E'].shape == tree.E.shape
    assert loaded_data['tree_s'].shape == tree.s.shape
