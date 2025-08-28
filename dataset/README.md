# Dataset Files Documentation

## Original Competition Data
- **`train.csv`** - Original Kaggle competition training data (36,696 rows)
- **`train_original.csv`** - Backup of original training data
- **`test.csv`** - Original Kaggle competition test data (3 rows, placeholder)
- **`sample_submission.csv`** - Sample submission format for Kaggle

## Synthetic Data - Original
- **`synth_original_367k_unbalanced.csv`** - LLM-generated synthetic dataset (366,960 rows)
  - Highly imbalanced: 40.3% True_Correct, 0.6% False_Correct
  - Contains: QuestionId, QuestionText, MC_Answer, StudentExplanation, Category, Misconception

## Balanced Datasets - Undersampled (2,330 samples per category)
- **`synth_undersampled_2330_per_cat.csv`** - Balanced dataset (13,980 rows total)
  - 6 categories × 2,330 samples = 16.7% each category
  - Undersampled to match smallest category (False_Correct)
  
### Train/Val/Test Splits (70/15/15)
- **`synth_undersampled_2330_per_cat_train_70pct.csv`** - Training set (9,786 rows)
- **`synth_undersampled_2330_per_cat_val_15pct.csv`** - Validation set (2,094 rows)  
- **`synth_undersampled_2330_per_cat_test_15pct.csv`** - Test set (2,100 rows)

## Balanced Datasets - Other Strategies
- **`synth_median_balanced_59k_per_cat.csv`** - Balanced to median category size (354,210 rows)
  - 6 categories × 59,035 samples each
  - Hybrid strategy: undersample large categories, oversample small ones
  
- **`synth_balanced_5000_per_cat.csv`** - Custom balanced dataset (30,000 rows)
  - 6 categories × 5,000 samples each
  - Good size for experimentation

## Weighted Dataset
- **`synth_original_with_inverse_freq_weights.csv`** - Original data with sample weights (366,960 rows)
  - Contains additional `sample_weight` column
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

## File Sizes
- Small (< 10 MB): Undersampled datasets, good for quick experiments
- Medium (30 MB): 5000 per category dataset
- Large (70+ MB): Median balanced and weighted datasets, for production training