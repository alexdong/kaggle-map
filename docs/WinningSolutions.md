Below is a synthesis of the main techniques, gains, failures and beginner tips described in the public winner write‑ups for two previous Kaggle competitions that are similar to *MAP – Charting Student Math Misunderstandings*.  Where possible, claims are referenced back to the original posts so you can investigate further.

### 1. Eedi – Mining Misconceptions in Mathematics (2024)

Only a few of the Eedi write‑ups were accessible without logging in; the second‑place solution was the most detailed.  The first‑place team used a two‑stage retrieval–rerank pipeline and large instruction‑tuned LLMs, but their full report requires authentication.

#### 2nd place solution – Team CQYR

**What worked**

* **Large retrievers & rerankers:**  Using increasingly larger Qwen models improved retrieval; a Linq‑Embeddings‑Mistral retriever scored only 0.461 on the private leaderboard, whereas a Qwen2.5‑32B retriever reached 0.495 with a 0.536 public LB.  The most powerful QwQ‑32B preview retriever delivered 0.500/0.531 (private/public).  Their best submission ensembled Mistral with two Qwen2.5‑14B models and fed the candidates into a 72B reranker, reaching 0.513/0.530.
* **Massive rerankers:**  An ablation study shows that adding a 72‑b‑parameter reranker and a Llama‑70B reranker to the baseline retriever boosted the private/public LB scores from 0.513/0.530 to 0.604/0.622—an absolute improvement of \~11 percentage points.  The authors concluded that “larger models tend to perform better” for reranking.
* **Synthetic questions & misconception augmentation:**  They generated synthetic MC questions and used LLMs to produce detailed explanations for each misconception.  These augmentations improved retriever performance by about 2–4 %.
* **Chain‑of‑thought (CoT) and pooling:**  Adding CoT to prompts improved retrieval and reranking but doubled inference time; last‑token pooling was found to outperform mean pooling.
* **Sliding‑window reranking & test‑time augmentation:**  Candidates were reranked in overlapping windows of 10 (stride = 7).  Test‑time augmentation by reversing prompt order and averaging scores gave a slight boost.
* **Quantisation & caching:**  They used `intel/auto‑round` quantisation to reduce model size with minimal accuracy loss and enabled `enabling_prefix_cache` in vLLM to cut inference time by \~10 %.

**What didn’t work**

* Smaller retrievers and rerankers performed poorly (e.g., Linq‑Embeddings‑Mistral at 0.461 private LB).
* Using BERT‑style models for reranking was inferior to large LLMs.
* Adding CoT to 14B and 32B models doubled inference time, so it had to be used sparingly.

**Tips/tricks**

* Use synthetic data and misconception augmentation to enrich training data; but filter questions with a quality score (they used GPT‑4o‑mini).
* Invest in the largest models you can afford, but be mindful of inference time; quantisation and prefix caching helped them deploy 72B models offline.
* Slide‑window reranking outperformed reranking all candidates at once.

#### Other accessible Eedi posts

The sixth‑place solution emphasised cross‑validation and balanced public/private splits but offered no technical details.  The seventh‑place (public 2nd) team used a voting ensemble of three retrieval–rerank pipelines, and the eighth‑place solution combined Qwen2.5 and Qwen‑32B models with a weighted ensemble.  Details of the first‑place “Magic Boost” and fifth‑place solutions require login and were not accessible.

### 2. KAChallenges Series 1 – Classifying Math Problems (2025)

This competition tasked participants with classifying math questions into one of eight categories.  The accessible winner write‑ups highlight several lessons for newcomers.

#### 1st place solution – *ducnh279*

**What worked**

* **Two base models and K‑fold training:**  The author fine‑tuned Qwen2.5‑14B and Qwen3‑14B for classification.  Each model was trained with 3‑fold stratified cross‑validation to maintain class balance.
* **Custom classification head:**  They replaced the Qwen3‑14B language‑model head with a linear classification head matching the embedding size.
* **Learning‑rate tuning and class weights:**  A smaller Qwen2.5‑0.5B model was used to search for optimal learning rates, and cosine‑decay scheduling without warm‑up worked best.  Computing class weights and adding them to the cross‑entropy loss helped the model focus on rare categories.
* **Weighted ensemble:**  Averaging the K‑fold models for each base model and then ensembling Qwen3‑14B (60 % weight) with Qwen2.5‑14B (40 % weight) improved the public LB to 0.9168 and the private LB to 0.9253, compared with CV scores around 0.909–0.911 for individual models.
* **Token truncation:**  Inputs were dynamically truncated to 300 tokens during training; the full question text was used at inference to maximise information.

**Tips/tricks**

* Use StratifiedKFold when classes are imbalanced.
* When using multiple GPUs, distributed data parallel (DDP) can speed up training.
* Weighting ensembles based on CV performance can squeeze out extra leaderboard points.

#### 2nd place solution – *Taha Alshatiri*

**What didn’t work**

* **Training on pseudo‑test labels alone:**  Finetuning only on pseudo test labels and then using the model on part of the training data degraded performance.
* **Tokenisation and embedding tricks:**  Pre‑processing techniques from other competitions (treating all numbers as one token, adding unknown tokens, removing LaTeX) hurt accuracy.
* **Large BERT models:**  DeBERTa and related BERT models only pushed the public LB into the 0.88s and could not surpass large Qwen models.

**What worked**

* **“Dirty” Qwen ensembling:**  The author created pseudo labels from public BERT notebooks and trained several Qwen models (Qwen‑3‑14B with and without augmentation, Qwen‑3‑32B with augmentation, Qwen‑2.5‑14B) for a single epoch.  They then combined predictions by hard voting, prioritising the 32B model.
* **Cosine learning‑rate schedule:**  Switching to a cosine scheduler for the final Qwen 2.5 model improved performance.
* **Resource tricks:**  Without a GPU for part of the competition, the author used Kaggle’s free L4 GPUs and later subscribed to Colab Pro to train heavier models.  They shared links to Colab notebooks and data augmentation datasets for reproducibility.

**Tips/tricks**

* Use pseudo labels from strong public notebooks to bootstrap training.
* Training each model for a single epoch can be sufficient when using high‑capacity models.
* When resources are limited, renting short‑term compute (e.g., Colab Pro) can make a big difference.

#### 3rd place solution – *Attacker*

* The author spent only \~5 hours on the competition and initially tried a reranker using a BAi/BGE model, which performed poorly.  They switched to a **LLM logit processor**, using Qwen3 and Qwen2.5 to classify problems directly via a prompt; Qwen2.5 had better baseline accuracy, so they fine‑tuned it with a 1e‑3 learning rate on \~1,280 samples and quantised the model with GPTQ.
* Their prompt asked the model to output only the category number and listed all eight categories in the instruction.
* Planned experiments—extracting only the answer through reasoning with Qwen3—were not completed due to time constraints.

**Tips/tricks**

* Even a simple Qwen2.5 model, fine‑tuned for a few hours, can achieve a top‑three finish.
* Quantising with GPTQ allowed them to deploy the model offline.

#### 4th place solution – *A. Abbas & Abdulkareem Omer*

* This team assembled **six Qwen models** (Qwen3 1.7B, Qwen3 0.6B, Qwen2.5Math 7B, Qwen2.5Math 1.5B, etc.) and fine‑tuned each via LoRA with slight variations in learning rate and number of epochs.
* They used four folds for CV and averaged predictions across folds.  The final ensemble blended all models with a weighted vote, producing a score good enough for fourth place.
* They emphasised that cleaning HTML and links was sufficient—no heavy augmentation was needed.

### General take‑aways for Kaggle newcomers

1. **Start with cross‑validation:**  Use appropriate split strategies (e.g., StratifiedKFold, GroupKFold) to get a reliable CV estimate; this guides model selection better than chasing the public leaderboard.
2. **Model size matters:**  Larger language models often yield higher retrieval/ranking accuracy but require careful handling (quantisation, caching, sliding‑window inference).
3. **Synthetic data helps:**  Generating synthetic questions and augmenting misconception explanations improved retrieval performance by a few percentage points.
4. **Don’t blindly copy tricks:**  Pre‑processing tricks from other competitions (tokenising numbers, removing LaTeX) may hurt performance.
5. **Efficient training:**  Parameter‑efficient fine‑tuning (LoRA), short training epochs and cosine learning‑rate schedules can produce strong models quickly.
6. **Practical hacks:**  Use offline notebooks and quantisation for large models; pay attention to inference time limits; renting short‑term compute can be cost‑effective; and always read Kaggle’s competition rules about code sharing and external data.

These summaries should give you a head start in understanding what top teams tried, what boosted their scores, what failed, and how you can approach your first Kaggle competition.

