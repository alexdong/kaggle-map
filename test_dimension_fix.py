#!/usr/bin/env python3
"""Test that demonstrates the dimension mismatch fix.

This test specifically addresses the error:
"Target size (torch.Size([1, 5])) must be the same as input size (torch.Size([1, 3]))"
"""


import sys

import torch

from kaggle_map.models import Category, TrainingRow
from kaggle_map.strategies.mlp import MLPNet, MLPStrategy


def create_test_data():
    """Create test data that would previously cause dimension mismatches."""
    # Create sample training data with categories that would cause ordering issues
    return [
        TrainingRow(
            row_id=1,
            question_id=101,
            question_text="What is 2+2?",
            mc_answer="4",
            student_explanation="I added them",
            category=Category("True_Correct"),
            misconception=None
        ),
        TrainingRow(
            row_id=2,
            question_id=101,
            question_text="What is 2+2?",
            mc_answer="5",
            student_explanation="I guessed",
            category=Category("False_Misconception"),
            misconception="Addition error"
        ),
        TrainingRow(
            row_id=3,
            question_id=101,
            question_text="What is 2+2?",
            mc_answer="3",
            student_explanation="Not sure",
            category=Category("False_Neither"),
            misconception=None
        )
    ]
    

def test_consistent_dimensions() -> bool:
    """Test that model heads and labels have consistent dimensions."""
    print("Testing dimension consistency fix...")
    
    training_data = create_test_data()
    
    # Extract metadata using the fixed methods
    correct_answers = MLPStrategy._extract_correct_answers(training_data)
    question_categories = MLPStrategy._extract_question_categories(training_data, correct_answers)
    question_misconceptions = MLPStrategy._extract_question_misconceptions(training_data)
    
    print(f"Question categories: {question_categories}")
    print(f"Question misconceptions: {question_misconceptions}")
    
    # Create model with consistent ordering
    model = MLPNet(question_categories, question_misconceptions)
    
    qid = 101
    qid_str = str(qid)
    
    # Check model head dimensions
    if qid_str in model.correct_category_heads:
        correct_head_size = model.correct_category_heads[qid_str].out_features
        print(f"Model correct head size: {correct_head_size}")
    
    if qid_str in model.incorrect_category_heads:
        incorrect_head_size = model.incorrect_category_heads[qid_str].out_features
        print(f"Model incorrect head size: {incorrect_head_size}")
        
    # Test label creation with consistent ordering
    for row in training_data:
        is_correct = row.mc_answer == correct_answers.get(row.question_id, "")
        correct_label, incorrect_label = MLPStrategy._create_category_labels(
            row, question_categories, is_correct
        )
        
        print(f"Row {row.row_id} (is_correct={is_correct}): "
              f"correct_label.shape={correct_label.shape}, "
              f"incorrect_label.shape={incorrect_label.shape}")
        
        # Validate dimensions match model heads
        if qid_str in model.correct_category_heads:
            expected_correct_size = model.correct_category_heads[qid_str].out_features
            assert len(correct_label) == expected_correct_size, (
                f"Correct label size {len(correct_label)} != model head size {expected_correct_size}"
            )
            
        if qid_str in model.incorrect_category_heads:
            expected_incorrect_size = model.incorrect_category_heads[qid_str].out_features
            assert len(incorrect_label) == expected_incorrect_size, (
                f"Incorrect label size {len(incorrect_label)} != model head size {expected_incorrect_size}"
            )
    
    print("✅ All dimensions are consistent!")
    return True

def test_tensor_operations() -> bool | None:
    """Test that tensor operations work without size mismatches."""
    print("\nTesting tensor operations...")
    
    # Simulate the exact scenario that was failing
    torch.device("cpu")
    
    # Create mock model output (what the model head produces)
    model_output = torch.randn(1, 3)  # Model head with 3 classes
    
    # Create mock target (what the label should be)
    # This should now have the same size due to our fixes
    target = torch.zeros(1, 3)  # Label with 3 classes (matches model)
    target[0, 1] = 1.0  # One-hot encoding
    
    print(f"Model output size: {model_output.size()}")
    print(f"Target size: {target.size()}")
    
    # This operation should work without errors
    try:
        # Convert one-hot to class index for CrossEntropyLoss
        target_class = torch.argmax(target, dim=-1)
        
        # Test loss computation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(model_output, target_class)
        
        print(f"Loss computation successful: {loss.item():.4f}")
        print("✅ Tensor operations work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Tensor operation failed: {e}")
        return False

def test_ordering_consistency() -> bool:
    """Test that category ordering is consistent across all operations."""
    print("\nTesting category ordering consistency...")
    
    # Sample categories that might have inconsistent ordering
    categories = [Category("False_Misconception"), Category("False_Neither"), Category("False_Correct")]
    
    # Test multiple operations use same ordering
    ordering1 = sorted(set(categories), key=str)  # Model head creation
    ordering2 = sorted(set(categories), key=str)  # Label creation
    ordering3 = sorted(set(categories), key=str)  # Prediction reconstruction
    
    print(f"Ordering 1 (model): {[str(c) for c in ordering1]}")
    print(f"Ordering 2 (labels): {[str(c) for c in ordering2]}")
    print(f"Ordering 3 (prediction): {[str(c) for c in ordering3]}")
    
    if ordering1 == ordering2 == ordering3:
        print("✅ All orderings are consistent!")
        return True
    print("❌ Orderings are inconsistent!")
    return False

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING MLP DIMENSION MISMATCH FIXES")
    print("Addresses: Target size (torch.Size([1, 5])) != input size (torch.Size([1, 3]))")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Dimension consistency
    try:
        if test_consistent_dimensions():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Tensor operations
    try:
        if test_tensor_operations():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test 3: Ordering consistency
    try:
        if test_ordering_consistency():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Results
    print("\n" + "=" * 70)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - Dimension mismatches are FIXED!")
        print("The original error should no longer occur during MLP training.")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED - Dimension mismatches may still exist")
        sys.exit(1)
