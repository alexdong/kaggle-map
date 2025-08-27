# MLP Neural Network Strategy Implementation Plan

Implement a question-specific MLP for **misconception detection**.

**Shared Trunk**: 
   - Input: embedding(384) 
   - Hidden 1: 768 units, ReLU, 0.3 dropout
   - Hidden 2: 384 units, ReLU, 0.3 dropout
   - Shared: 192 units, ReLU, 0.3 dropout

**Question-Specific Misconception Heads** (`nn.ModuleDict`): 
   - Q31772: 3 outputs → [Incomplete, WNB, NA]
   - Q31774: 4 outputs → [Mult, SwapDividend, FlipChange, NA]  
   - Q31778: 4 outputs → [Additive, Irrelevant, WNB, NA]
   - Q32829: 4 outputs → [Not_variable, Adding_terms, Inverse_operation, NA]
   - ... (one head per question with question-specific misconceptions + NA)


- Follow existing `Strategy` pattern from `kaggle_map/strategies/base.py`
- Use data structures `TrainingInput` for X and Label above, where
   **X**: `"Question: {}; Expected Answer: {}; Student's Answer: {}; Student's Explanation: {}"` ->Embedding (384-dim)
   **label**: Misconception | "NA"
- Use 70% train.csv for training, 15% for validation and 15% for testing
- Embedding calculated using `core/embeddings/tokenizer.py`

## Loss Function Design
[ ] **Misconception Loss**: `BCEWithLogitsLoss` for question-specific multi-label prediction
[ ] **Training**: Each question's head trained on all samples for that question
[ ] **Loss aggregation**: Average loss across all question heads