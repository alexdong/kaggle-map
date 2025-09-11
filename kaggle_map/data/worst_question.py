"""Find the worst performing question by analyzing error rates and prediction accuracy."""

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

logger.info("Loading datasets...")
train_df = pd.read_csv("datasets/train.csv")
error_df = pd.read_csv("datasets/error_prediction.csv")

logger.info(f"Train dataset: {len(train_df)} rows")
logger.info(f"Error dataset: {len(error_df)} rows")

# Analyze each question's performance
question_stats = []

for qid in train_df["QuestionId"].unique():
    # Get all rows for this question from both datasets
    train_q = train_df[train_df["QuestionId"] == qid]
    error_q = error_df[error_df["QuestionId"] == qid]

    # Total samples in training data for this question
    total_samples = len(train_q)

    # Samples that appear in error prediction (these are the ones MLP got wrong)
    error_samples = len(error_q)

    # Calculate the actual error rate: errors / total
    error_rate = error_samples / total_samples * 100 if total_samples > 0 else 0

    # Within the errors, count TRUE vs FALSE predictions
    if not error_q.empty:
        true_predictions = (error_q["Category"].str.contains("TRUE")).sum()
        false_predictions = (error_q["Category"].str.contains("FALSE")).sum()
        # This is accuracy WITHIN the errors (how well we identify misconceptions for wrong answers)
        misconception_accuracy = true_predictions / len(error_q) * 100 if len(error_q) > 0 else 0

        # Get unique misconceptions for this question
        unique_misconceptions = error_q["actual_misconception"].nunique()

        # Get the question text (first occurrence)
        question_text = str(error_q.iloc[0]["QuestionText"])[:80]
        if len(str(error_q.iloc[0]["QuestionText"])) > 80:
            question_text += "..."

        # Get correct answer
        correct_answer = str(error_q.iloc[0].get("CorrectAnswer", "N/A"))

        # Most common wrong answer
        wrong_answers = error_q["MC_Answer"].value_counts()
        most_common_wrong = wrong_answers.index[0] if not wrong_answers.empty else "N/A"

    else:
        true_predictions = 0
        false_predictions = 0
        misconception_accuracy = 0.0  # No errors to classify
        unique_misconceptions = 0
        question_text = str(train_q.iloc[0]["QuestionText"])[:80] if not train_q.empty else "N/A"
        correct_answer = "N/A"
        most_common_wrong = "N/A"

    # Calculate overall model accuracy for this question (correct answers / total)
    model_accuracy = (total_samples - error_samples) / total_samples * 100 if total_samples > 0 else 100

    question_stats.append({
        "QuestionId": qid,
        "QuestionText": question_text,
        "TotalSamples": total_samples,
        "ErrorSamples": error_samples,
        "ErrorRate": error_rate,
        "ModelAccuracy": model_accuracy,
        "TruePredictions": true_predictions,
        "FalsePredictions": false_predictions,
        "MisconceptionAccuracy": misconception_accuracy,
        "UniqueMisconceptions": unique_misconceptions,
        "CorrectAnswer": correct_answer,
        "MostCommonWrong": most_common_wrong,
    })

# Convert to DataFrame for easier analysis
stats_df = pd.DataFrame(question_stats)

# Sort by error rate (worst first) - this is the key metric
stats_df = stats_df.sort_values(by=["ErrorRate", "ErrorSamples"], ascending=[False, True])

# Display results
console = Console()
table = Table(title="Question Performance Analysis (Worst Error Rate First)")

# Add columns
table.add_column("Q#", style="cyan", no_wrap=True)
table.add_column("Question", style="blue", overflow="fold", max_width=40)
table.add_column("Error\nRate", style="red", justify="right")
table.add_column("Model\nAcc", style="yellow", justify="right")
table.add_column("Total", style="white", justify="right")
table.add_column("Errors", style="magenta", justify="right")
table.add_column("Misc\nAcc", style="green", justify="right")
table.add_column("TRUE", style="green", justify="right")
table.add_column("FALSE", style="red", justify="right")
table.add_column("Correct Answer", style="green", overflow="fold")
table.add_column("Common Wrong", style="yellow", overflow="fold")

# Add rows
for _, row in stats_df.iterrows():
    # Color code based on error rate
    error_color = "red" if row["ErrorRate"] > 15 else "yellow" if row["ErrorRate"] > 10 else "green"
    model_color = "red" if row["ModelAccuracy"] < 85 else "yellow" if row["ModelAccuracy"] < 90 else "green"

    table.add_row(
        f"Q{row['QuestionId']}",
        row["QuestionText"],
        f"[{error_color}]{row['ErrorRate']:.1f}%[/{error_color}]",
        f"[{model_color}]{row['ModelAccuracy']:.1f}%[/{model_color}]",
        str(row["TotalSamples"]),
        str(row["ErrorSamples"]),
        f"{row['MisconceptionAccuracy']:.1f}%",
        str(row["TruePredictions"]),
        str(row["FalsePredictions"]),
        row["CorrectAnswer"],
        row["MostCommonWrong"]
    )

console.print("\n")
console.print(table)

# Summary statistics
logger.info("\n=== Summary Statistics ===")
worst_question = stats_df.iloc[0]
logger.info(f"Worst performing question: Q{worst_question['QuestionId']}")
logger.info(f"  Error rate: {worst_question['ErrorRate']:.1f}% ({worst_question['ErrorSamples']}/{worst_question['TotalSamples']} samples)")
logger.info(f"  Model accuracy: {worst_question['ModelAccuracy']:.1f}%")
logger.info(f"  Misconception identification: {worst_question['MisconceptionAccuracy']:.1f}% accurate")
logger.info(f"  FALSE predictions: {worst_question['FalsePredictions']} ({worst_question['FalsePredictions']/worst_question['ErrorSamples']*100:.1f}% of errors)")
logger.info(f"  Question: {worst_question['QuestionText']}")

# Questions with highest error rates (min 100 total samples)
high_volume = stats_df[stats_df["TotalSamples"] >= 100]
if not high_volume.empty:
    logger.info("\n=== Worst Error Rates (100+ samples) ===")
    worst_high_volume = high_volume.nsmallest(3, "ModelAccuracy")
    for _, q in worst_high_volume.iterrows():
        logger.info(f"Q{q['QuestionId']}: {q['ErrorRate']:.1f}% error rate ({q['ErrorSamples']}/{q['TotalSamples']} samples)")

# Questions with most absolute errors
logger.info("\n=== Most Absolute Errors ===")
most_errors = stats_df.nlargest(3, "ErrorSamples")
for _, q in most_errors.iterrows():
    logger.info(f"Q{q['QuestionId']}: {q['ErrorSamples']} errors out of {q['TotalSamples']} ({q['ErrorRate']:.1f}% error rate)")

logger.info(f"\n✅ RECOMMENDATION: Focus on Question {worst_question['QuestionId']} - it has the worst error rate at {worst_question['ErrorRate']:.1f}%")
