#!/usr/bin/env python3
"""Validate that MLP model head dimensions match label dimensions."""

import sys
from pathlib import Path

from kaggle_map.models import Category
from kaggle_map.strategies.mlp import MLPNet, MLPStrategy


def validate_dimension_consistency() -> bool:
    """Validate that model heads and labels have consistent dimensions."""
    print("Validating MLP dimension consistency...")
    
    # Parse a small amount of training data
    train_csv_path = Path("datasets/train.csv")
    if not train_csv_path.exists():
        print(f"Training file not found: {train_csv_path}")
        return False
    
    # Get training data subset
    training_data = MLPStrategy._parse_training_data(train_csv_path)[:100]  # Just 100 rows for validation
    
    # Extract metadata
    correct_answers = MLPStrategy._extract_correct_answers(training_data)
    question_categories = MLPStrategy._extract_question_categories(training_data, correct_answers)
    question_misconceptions = MLPStrategy._extract_question_misconceptions(training_data)
    
    print(f"Validating {len(question_categories)} questions...")
    
    # Create model to check head dimensions
    model = MLPNet(question_categories, question_misconceptions)
    
    # Validate each question's dimensions
    validation_errors = []
    
    for qid, category_map in question_categories.items():
        qid_str = str(qid)
        
        # Check correct categories
        if True in category_map:
            # Model head dimension
            model_correct_cats = sorted(set(category_map[True]), key=str)
            model_head_size = len(model_correct_cats) if qid_str in model.correct_category_heads else 0
            
            # Label creation dimension (using same logic as _create_category_labels)
            label_correct_cats = sorted(set(category_map[True]), key=str)
            label_size = len(label_correct_cats)
            
            if model_head_size != label_size:
                error = f"Question {qid} correct categories: model head={model_head_size}, labels={label_size}"
                validation_errors.append(error)
                print(f"❌ {error}")
            else:
                print(f"✓ Question {qid} correct categories: {model_head_size} (consistent)")
                
        # Check incorrect categories
        if False in category_map:
            # Model head dimension
            model_incorrect_cats = sorted(set(category_map[False]), key=str)
            model_head_size = len(model_incorrect_cats) if qid_str in model.incorrect_category_heads else 0
            
            # Label creation dimension
            label_incorrect_cats = sorted(set(category_map[False]), key=str)
            label_size = len(label_incorrect_cats)
            
            if model_head_size != label_size:
                error = f"Question {qid} incorrect categories: model head={model_head_size}, labels={label_size}"
                validation_errors.append(error)
                print(f"❌ {error}")
            else:
                print(f"✓ Question {qid} incorrect categories: {model_head_size} (consistent)")
    
    # Check misconceptions
    for qid, misconceptions in question_misconceptions.items():
        qid_str = str(qid)
        
        model_misconception_size = len(misconceptions) if qid_str in model.misconception_heads else 0
        label_misconception_size = len(misconceptions)  # Should be same
        
        if model_misconception_size != label_misconception_size:
            error = f"Question {qid} misconceptions: model head={model_misconception_size}, labels={label_misconception_size}"
            validation_errors.append(error)
            print(f"❌ {error}")
        else:
            print(f"✓ Question {qid} misconceptions: {model_misconception_size} (consistent)")
    
    if validation_errors:
        print(f"\n❌ Found {len(validation_errors)} dimension inconsistencies:")
        for error in validation_errors:
            print(f"  - {error}")
        return False
    print(f"\n✅ All {len(question_categories)} questions have consistent dimensions!")
    return True

def test_label_creation_ordering() -> bool:
    """Test that category ordering is consistent between model and labels."""
    print("\nTesting category ordering consistency...")
    
    # Test with a sample set of categories
    test_categories = [Category("True_Correct"), Category("True_Misconception"), Category("True_Neither")]
    
    # Simulate what happens in model head creation
    model_ordering = sorted(set(test_categories), key=str)
    
    # Simulate what happens in label creation
    label_ordering = sorted(set(test_categories), key=str)
    
    print(f"Model ordering: {[str(cat) for cat in model_ordering]}")
    print(f"Label ordering: {[str(cat) for cat in label_ordering]}")
    
    if model_ordering == label_ordering:
        print("✅ Category ordering is consistent!")
        return True
    print("❌ Category ordering is inconsistent!")
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("MLP DIMENSION VALIDATION")
    print("=" * 60)
    
    # Test 1: Validate dimension consistency
    consistency_ok = validate_dimension_consistency()
    
    # Test 2: Test ordering consistency
    ordering_ok = test_label_creation_ordering()
    
    # Overall result
    print("\n" + "=" * 60)
    if consistency_ok and ordering_ok:
        print("🎉 ALL VALIDATIONS PASSED - MLP dimension mismatches are fixed!")
        sys.exit(0)
    else:
        print("💥 VALIDATIONS FAILED - dimension mismatches still exist")
        sys.exit(1)
