# Support ticket triage

This classifies a customer support message into one of four routes, account-access, transaction-dispute, fraud-report, general. The model is TF-IDF with unigrams plus logistic regression with balanced class weights. Macro F1 is 0.972 from nested cross validation. Fraud-report recall is 1.000.

## Setup
uv can be installed from https://docs.astral.sh/uv/ and the project needs Python 3.13. uv will fetch it if it is missing.
```bash
uv sync
```

## Predict one message
```python
from models.predict import predict
predict("someone moved my coins to a wallet I don't recognise")
```

## Score a holdout CSV
The trained model is committed at artifacts/model.joblib, so there is no need to train anything first.
The input CSV needs a column named text, any other columns are passed through untouched, and the output is the same rows with a predicted_label column added.

### Locally
```bash
uv sync
uv run python scripts/score_holdout.py your_holdout.csv predictions.csv
```
It prints the row count and the output path when it finishes.

### With Docker
The image is at https://hub.docker.com/r/midhileshmomidi489/coinbase-assessment and needs no local Python or uv. The folder holding the CSV must be mounted so the output comes back to the host.
```bash
docker run --rm -v "$(pwd)":/work \
  midhileshmomidi489/coinbase-assessment:latest \
  python scripts/score_holdout.py /work/your_holdout.csv /work/predictions.csv
```
On Windows PowerShell use ${PWD} instead of $(pwd).
On Linux, if the write fails with a permission error, add --user $(id -u):$(id -g) because the image runs as a non-root user.
Running the image with no arguments scores the bundled sample file and prints the row count, which is what CI uses as a smoke test.

## Tests
```bash
uv run pytest tests --cov --cov-report=term-missing
uv run ruff check .
```

There are 33 tests and coverage is enforced at 90 percent over src. plots.py and tuning.py are excluded because plotting has nothing useful to assert and the grid search fits hundreds of models.

## Reproduce the analysis
```bash
uv run python scripts/run_analysis.py      # class balance, duplication, plots
uv run python scripts/train.py             # cross validation for all three pipelines
uv run python scripts/tune.py              # grid search and nested CV
uv run python scripts/run_diagnostics.py   # learning curve, McNemar, novel messages
```

Everything is seeded with random_state 42.

## Data leakage
The 400 rows are templated. After normalising greetings, sign-offs, numbers and asset tickers they collapse to 210 unique templates. That is a 47.5 percent duplication rate. Up to 9 rows share one template. No template appears under two labels.

A random split puts near identical messages on both sides and reports about 0.97 macro F1 that will not hold on a hidden holdout. So every split uses StratifiedGroupKFold, stratified on label and grouped on the template key. No template appears on both sides of any fold and there is a test that asserts this.

This makes the numbers lower and honest. The hidden holdout will probably score higher if it comes from the same templates.

## Metric
fraud-report is the rarest class at 50 of 400, which is 12.5 percent, and the most expensive to get wrong. A missed fraud report sits in a queue while money moves. A false fraud alarm costs an agent thirty seconds. So fraud-report recall is the primary guardrail and macro F1 is the overall health check.

Accuracy is not used because the baseline scored 96 percent accuracy while missing 8 percent of fraud tickets. ROC-AUC was computed and rejected because out of fold macro OvR is 0.9988 and fraud-report is 1.0000, so it cannot tell the three models apart, and it averages over thresholds that are never used since the service returns an argmax.

## Class imbalance
The ratio is 3.2 to 1 between general at 160 rows and fraud-report at 50. Handled with class_weight balanced. No resampling and no SMOTE.

The unweighted baseline missed 4 of 50 fraud tickets and all four went to general, not to transaction-dispute. Errors going to the majority class is a class prior problem, not a signal problem.

To catch this in production you track per class recall, not accuracy, and watch the share of tickets predicted fraud-report. If it drifts below the 12.5 percent base rate the prior is winning again.

## Experiments
| pipeline | macro F1 | fold std | fraud recall | errors |
|---|---|---|---|---|
| tfidf_logreg | 0.9544 | 0.043 | 0.920 | 16 |
| tfidf_logreg_balanced | 0.9627 | 0.062 | 1.000 | 14 |
| tfidf_logreg_tuned | 0.9862 | 0.020 | 1.000 | 6 |

 tfidf_logreg_tuned is the shipped model. The full log is in reports/experiments.csv.

A 30 point grid over ngram_range, C and sublinear_tf is in reports/tuning.md. Unigrams beat bigrams and trigrams in every configuration. Bigrams learn template phrasing like `stuck in pending` as a unit and that does not transfer to an unseen template. Dropping them cut transaction-dispute to account-access errors from 8 to 2 and cut fold variance by three times.

The grid was selected on the same folds it was scored on, so nested cross validation gives the unbiased number, 0.972 plus or minus 0.043. All five outer folds picked unigrams on their own. The headline number is 0.972, not 0.9862.

McNemar results from reports/diagnostics.md:
- baseline to balanced, 4 fixed and 2 broken, p = 0.6875, not significant, kept anyway because it moves errors off the expensive class
- balanced to tuned, 8 fixed and 0 broken, p = 0.0078, significant

## Diagnostics
Learning curve gap goes 0.30, 0.095, 0.055, 0.015 and validation is still rising at 100 percent of the data, so the model is short on data rather than overfitting.

10 hand written messages using words not in the training set, 10 correct.

## Scope and trade-offs
What was prioritised: finding the template leakage and building a split that survives it, a metric argument based on the cost of the fraud route, error analysis on every wrong ticket, tuning checked with nested CV, significance tests instead of eyeballing deltas.

What was built beyond the required core: a Dockerfile and a CI pipeline. The reason is that Docker is not installed locally, so CI is how the image gets built and actually run on every push. The workflow runs lint, then tests with a coverage floor, then builds the image, runs it, and only then pushes to Docker Hub. Containerising found a real bug: once the package is installed instead of run from source, PROJECT_ROOT resolved inside the virtualenv instead of the working directory and the model file could not be found. That only shows up in a container.

What was left out and why:
- A FastAPI wrapper. predict(text) and the batch script are the two interfaces this needs. An HTTP layer adds surface area but no evidence about the model.
- An LLM comparison. The remaining 6 errors come from two held out templates, and at 400 rows even a two error difference is not significant, so the comparison would produce a number that cannot be defended.

With more time: more varied templates rather than a bigger model, because the learning curve says data is the limit. Then a confidence threshold for escalation. At 0.5, escalating 23 percent of traffic catches every error the model makes.

At 10,000 requests per minute this model is about 1ms of CPU per request and needs one modest box. The reason to prefer it over an LLM is not cost, it is latency variance and determinism, since it cannot return a label outside the four routes. An LLM is the right call for a fifth route with no labelled data, or when the agent needs a reason along with the label.

Time spent: roughly five focused hours across a day.
