# Reflection — Customer Intelligence Platform

*Answers to the section 12 prompts from the project brief.*

---

## Why this model family and this threshold over the alternatives?

I chose XGBoost as the improved model over Random Forest and neural approaches for three reasons. First, it trains significantly faster on tabular data of this size, which mattered during iteration. Second, it handles the class imbalance in the bank marketing dataset better through the `scale_pos_weight` parameter — the positive class (subscription) is only about 12% of the data. Third, it consistently hit the PR-AUC promotion gate threshold in testing while Random Forest only cleared it marginally and was sensitive to random seed.

The threshold of 3 PR-AUC percentage points over the baseline was chosen by running the improved model 5 times with different random seeds and observing the variance. The improvement was consistently between 4 and 7 points, so 3pp gives a buffer against noise without being so lenient it lets a mediocre model through. Logistic Regression baseline sits around 0.38 PR-AUC on this dataset; XGBoost reliably reaches 0.44 to 0.47, which is a meaningful business difference when scoring thousands of customers for campaign outreach.

---

## What broke first when you tried to deploy, and what did you change?

The CFPB data download broke first. The API endpoint dropped the connection mid-stream consistently, giving a `ChunkedEncodingError`. The original code used a single `requests.get()` call with no streaming, which worked fine for the small bank marketing zip but failed on the larger complaint dataset.

I rewrote the download function with four fallback strategies: the JSON API with streaming, then the CSV API endpoint, then downloading the full zip file with an early cutoff at 50MB, then generating synthetic complaints if all three fail. This made the data pipeline robust enough to run in CI on a fresh machine regardless of CFPB API availability.

The second thing that broke was the nested zip in the UCI Bank Marketing download. The outer zip contains `bank.zip` rather than the CSV directly. The original code assumed a flat zip structure. Fixed by detecting the nested case and opening the inner zip before reading the CSV.

---

## Why your gate margin, and what fails if you tighten PR-AUC by another 2 points?

The gate is set at +3 PR-AUC points over the baseline. If tightened to +5 points, the improved model would be blocked on roughly 30% of runs due to variance from the train/test split. The same XGBoost model with the same hyperparameters, trained on the same data, can produce PR-AUC anywhere from 0.42 to 0.48 depending on which 20% of the data ends up in the test set. A +5pp gate would make the promotion decision noisy and unreliable — sometimes passing the same model, sometimes blocking it. That is worse than a slightly lenient threshold because it destroys trust in the gate itself.

A proper fix would be cross-validated PR-AUC (5-fold) as the gate metric rather than a single train/test split, which would reduce variance significantly. That was not implemented here due to time constraints.

---

## A complaint answer your RAG got wrong or ungrounded

During the 10-question eval, the question "What do customers say about cryptocurrency complaints?" produced a weak answer. The CFPB complaint corpus used here is filtered to Mortgage, Credit card, Checking accounts, and Student loans — cryptocurrency complaints are not in the index. The retriever returned chunks with similarity scores below the 0.35 threshold, so the system correctly refused to answer.

However, when the threshold was temporarily lowered to 0.25 for testing, the system retrieved loosely related complaints about "unauthorised transactions" and "account freezes" from banking products, and the LLM constructed a plausible-sounding but misleading answer about cryptocurrency complaints using that unrelated evidence. The cited IDs were real complaint records but they were about debit card fraud, not cryptocurrency. The eval caught this because the expected evidence IDs did not match — a clear FAIL.

This is the core danger of RAG systems: retrieval can find something, the LLM can make it sound coherent, and without a per-question ground truth eval you would never know it was wrong. The refusal threshold is the first line of defence; the eval harness is the second.

---

## The one risk not fully closed if this went live tomorrow

No authentication on any endpoint. Right now, any person or script that can reach the server can call `/predict` with any customer data and get a prediction, or call `/ask-complaints` with any question and consume Groq API quota. In a real financial services context this would be a serious data governance and cost control problem. A customer's predicted subscription probability is sensitive commercial information. The fix is straightforward — API key authentication via a FastAPI dependency — but it was not implemented in time for this submission.

---

## What a senior MLOps engineer would criticise first

The first thing they would open is `monitoring/ml_drift.py` and ask: "Where is the retrain trigger?" Detecting drift and generating a report is only half the loop. A production system needs an automated action — a GitHub Actions workflow dispatch, a webhook, an alert to a Slack channel — that fires when drift crosses a threshold. Right now the report sits in a folder and nothing happens with it. The monitoring is observational, not operational.

The second criticism would be the local MLflow setup. Experiment history lives on whoever's laptop ran the training. If that machine is reformatted, all run history, artifacts, and the model version that is currently in production are gone. A remote artifact store (Azure Blob Storage, S3, or even a shared network drive) would take two hours to set up and would make the whole system recoverable.
