# MAP – Charting Student Math Misunderstandings

## Competition Overview

The **MAP – Charting Student Math Misunderstandings** competition is a featured code competition on Kaggle hosted by **The Learning Agency LLC** with support from Vanderbilt University and the Eedi educational platform. 

### Mission
Build NLP models that predict the affinity between misconceptions and student open-ended responses. The goal is to detect and classify math misconceptions from real student explanations, helping teachers give faster, more targeted feedback and unlock new insights into how students learn.

### Why This Matters
When students answer diagnostic questions on Eedi, they sometimes explain their answer. These explanations often reveal misconceptions, but tagging explanations manually is time-consuming. The competition asks participants to create models that can automatically suggest likely misconceptions so teachers can address errors more effectively. Winning models will help teachers provide faster, more targeted feedback and unlock new insights into how students learn.


## Dataset and Tasks

### Data Description

The dataset consists of diagnostic questions from Eedi. After choosing a multiple-choice answer, students may provide a written explanation. Each row in the **train/test** CSV represents one student response and contains:

| Column | Description |
|--------|-------------|
| QuestionId | Unique identifier for the question |
| QuestionText | Text of the question. OCR has been applied to images so the text is available without processing the images |
| MC_Answer | The multiple-choice answer selected by the student |
| StudentExplanation | Free-text explanation given by the student |
| Category (train only) | Relationship between the selected answer and explanation. Possible values are True_Correct, True_Incorrect, True_Misconception (correct answer but explanation shows a misconception) and their False_* counterparts |
| Misconception (train only) | Specific math misconception tag (e.g., "Incomplete fraction simplification"); NA if no misconception applies |

### Submission Format

The **sample submission** file contains two columns:
- `row_id`: Identifier for each test row
- `predictions`: Up to three Category:Misconception combinations separated by spaces

Example submission:
```
row_id,predictions
0,True_Misconception:Incomplete True_Incorrect:Confused_operation True_Correct:NA
1,False_Correct:NA False_Misconception:Wrong_formula False_Incorrect:Calculation_error
```

**Important**: Predictions beyond the top three are ignored by the evaluation system.

### Target Tasks

Models must perform three sub-tasks:

1. **Determine if the multiple-choice answer is correct**
   - Decide whether the student's chosen answer is right or wrong
   - This is encoded in the Category label as True_* (correct) or False_* (incorrect)

2. **Determine whether the explanation reveals a misconception**
   - Some explanations show misunderstandings even when the answer is correct
   - Example: Student gets the right answer but explains it using incorrect reasoning
   - This influences whether the Category is labeled _Correct, _Incorrect, or _Misconception

3. **Identify the specific misconception tag**
   - When a misconception exists, identify the specific type
   - Examples: "Incomplete simplification", "Confused operation", "Wrong formula"
   - There is exactly one misconception label per explanation when applicable

### Additional Data Notes

- **Image Data**: Questions are typically displayed as images. Organisers used human-in-the-loop OCR to extract question text and provide it in QuestionText. The original images and bounding boxes are included for participants who want to work with images.

- **Limited Questions**: The train.csv includes only 15 unique questions. Kaggle staff member Chris Deotte confirmed via leaderboard probing that the test set does not contain any new questions; therefore participants can cross-validate using K-folds without a group split on question ID.

- **Real Student Data**: The dataset contains authentic student responses that may include spelling errors, incomplete thoughts, or ambiguous language, making this a realistic and challenging NLP task.

## Evaluation Metric (MAP@3)

Submissions are evaluated using **Mean Average Precision @ 3 (MAP@3)**. For each observation, participants may submit up to three predicted Category:Misconception pairs ranked by confidence.

### Understanding MAP@3

MAP@3 is a ranking quality metric that evaluates both:
1. **Relevance**: Whether the predicted misconceptions are correct
2. **Ranking**: Whether more relevant predictions appear higher in the list

### How MAP@3 Works

For each student response:
1. The model predicts up to 3 Category:Misconception pairs in ranked order
2. Average Precision (AP) is calculated based on where the correct answer appears:
   - If correct answer is at position 1: AP = 1.0
   - If correct answer is at position 2: AP = 0.5  
   - If correct answer is at position 3: AP ≈ 0.33
   - If correct answer is not in top 3: AP = 0.0
3. The MAP@3 score is the mean of all AP scores across the test set

### Score Range
- **Perfect score**: 1.0 (all correct answers at position 1)
- **Worst score**: 0.0 (no correct answers in top 3)
- **Random baseline**: ~0.28 (varies based on class distribution)

### Example Calculation

Given 3 test samples with ground truth and predictions:
```
Sample 1: Ground truth = "True_Misconception:Incomplete"
  Predictions: ["True_Misconception:Incomplete", "True_Incorrect:Other", "False_Correct:NA"]
  AP = 1.0 (correct at position 1)

Sample 2: Ground truth = "False_Correct:NA"
  Predictions: ["True_Incorrect:Wrong", "False_Correct:NA", "True_Misconception:Other"]
  AP = 0.5 (correct at position 2)

Sample 3: Ground truth = "True_Incorrect:Calculation"
  Predictions: ["False_Correct:NA", "True_Misconception:Other", "False_Incorrect:Wrong"]
  AP = 0.0 (correct answer not in top 3)

MAP@3 = (1.0 + 0.5 + 0.0) / 3 = 0.5
```

