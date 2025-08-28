# Dataset Files Documentation

## Original Competition Data
- **`train.csv`** - Original Kaggle competition training data (36,696 rows, 7.5 MB)
  - Contains: row_id, QuestionId, QuestionText, MC_Answer, StudentExplanation, Category, Misconception
- **`test.csv`** - Original Kaggle competition test data (3 rows, placeholder)
  - Contains: row_id, QuestionId, QuestionText, MC_Answer, StudentExplanation
- **`sample_submission.csv`** - Sample submission format for Kaggle (3 rows)
  - Format: row_id, predicted_categories

## Synthetic Data - Original
- **`synth_original_367k_unbalanced.csv`** - LLM-generated synthetic dataset (366,960 rows, 75.2 MB)
  - Highly imbalanced: 40.3% True_Correct, 0.6% False_Correct
  - Contains: row_id, QuestionId, QuestionText, MC_Answer, StudentExplanation, Category, Misconception
  - 7 columns total

## Balanced Datasets - Undersampled (2,330 samples per category)
- **`synth_undersampled_2330_per_cat.csv`** - Balanced dataset (13,980 rows, 2.8 MB)
  - 6 categories × 2,330 samples = 16.7% each category
  - Undersampled to match smallest category (False_Correct)
  
### Train/Val/Test Splits (70/15/15)
- **`synth_undersampled_2330_per_cat_train_70pct.csv`** - Training set (9,786 rows, 2.0 MB)
- **`synth_undersampled_2330_per_cat_val_15pct.csv`** - Validation set (2,094 rows, 0.4 MB)  
- **`synth_undersampled_2330_per_cat_test_15pct.csv`** - Test set (2,100 rows, 0.4 MB)

## Balanced Datasets - Other Strategies
- **`synth_median_balanced_59k_per_cat.csv`** - Balanced to median category size (354,210 rows, 70.6 MB)
  - 6 categories × 59,035 samples each
  - Hybrid strategy: undersample large categories, oversample small ones
  
- **`synth_balanced_5000_per_cat.csv`** - Custom balanced dataset (30,000 rows, 6.0 MB)
  - 6 categories × 5,000 samples each
  - Good size for experimentation

## Weighted Dataset
- **`synth_original_with_inverse_freq_weights.csv`** - Original data with sample weights (366,960 rows, 81.9 MB)
  - Contains additional `sample_weight` column (8 columns total)
  - Weights inversely proportional to class frequency
  - Use for weighted loss functions during training

## Category Distribution

All balanced datasets have these 6 categories equally distributed:
1. `True_Correct` - Correct answer, no misconception
2. `True_Neither` - Correct answer, unclear if misconception
3. `True_Misconception` - Correct answer despite misconception  
4. `False_Correct` - Wrong answer, correct reasoning
5. `False_Neither` - Wrong answer, unclear reasoning
6. `False_Misconception` - Wrong answer due to misconception

## File Sizes Summary
- **Small (< 10 MB)**: Undersampled datasets, good for quick experiments
  - synth_undersampled_2330_per_cat*.csv files
  - synth_balanced_5000_per_cat.csv
  - Original train.csv
- **Medium (30-75 MB)**: 
  - synth_median_balanced_59k_per_cat.csv
  - synth_original_367k_unbalanced.csv
- **Large (75+ MB)**: 
  - synth_original_with_inverse_freq_weights.csv

## Column Descriptions
- **row_id**: Unique identifier for each row
- **QuestionId**: Numeric ID for the math question
- **QuestionText**: The full text of the math problem
- **MC_Answer**: Student's multiple choice answer
- **StudentExplanation**: Student's written explanation 
- **Category**: One of 6 categories combining correctness and misconception status
- **Misconception**: Specific misconception ID (if applicable)
- **sample_weight**: Inverse frequency weight for balanced training (weighted dataset only)