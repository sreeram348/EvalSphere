# EvalSphere

EvalSphere is a central AI and machine-learning workspace for software quality
engineering. It brings prompt improvement, test-automation asset generation,
multimodal evidence analysis, and tabular model benchmarking into one Streamlit
app.

## Workflows

- **Prompt studio:** Build and edit structured prompts for XPath locators,
  Robot Framework automation, test cases, and custom QA tasks.
- **Model evaluation:** Send a prompt through the local FastAPI service, record
  the generated response and latency, and review recent runs from SQLite.
- **Image analysis:** Submit a screenshot with context to identify visible UI
  issues and generate evidence-based verification steps.
- **Audio analysis:** Transcribe a recording, then use the selected chat model
  to summarize evidence, defects, decisions, or action items.
- **ML benchmark:** Compare classification and regression models on labeled
  tables, or run unsupervised clustering without a target column. Data and
  training stay in the app process and are not sent to the generative model.

## ML benchmark inputs and evaluation

The benchmark accepts CSV, Excel (`.xlsx`), JSON, Parquet, HTML tables, XML
tables, and Markdown pipe tables. Excel workbooks and HTML files with multiple
tables let you select the worksheet or table to analyze. Uploads are limited
to 20 MB, 15,000 rows, and 100 usable features.

Classification compares nine CPU-friendly models, including logistic
regression, support vector machines, random forests, extra trees, gradient
boosting, and a scikit-learn multilayer perceptron. It ranks models by
weighted F1 and reports additional quality and runtime metrics. Optional
bounded cross-validation tuning adjusts the leading model using training
rows. Regression compares eight models and ranks by held-out R²; clustering
compares K-Means, Mini-Batch K-Means, and Gaussian Mixture using sampled
silhouette and related metrics.

Choose a target column for supervised classification or regression. For
clustering, select the input columns and exclude identifiers or known labels.
Model rankings describe performance on the uploaded dataset and its evaluation
split; they are not universal rankings or a guarantee of state-of-the-art
results. The leading model is selected from that split, so use the leaderboard
as a shortlist and confirm final choices on a fresh holdout or with nested
cross-validation.

Graph learning, NLP-specific tasks, web scraping, PDF ingestion, and video
analysis are separate workflows and are not part of this tabular benchmark.
The existing image and audio analysis workflows send content to the configured
provider only when you explicitly run model analysis; they do not train the
local ML models.

## Local model evaluation service

The evaluation workflow uses a small FastAPI service and SQLAlchemy-backed
SQLite store. Start both processes from the repository root in separate
terminals:

```powershell
python -m uvicorn evaluation_api:app --host 127.0.0.1 --port 8000
```

```powershell
streamlit run app.py
```

The Streamlit page sends one request to the selected configured mini model only
after you press **Run evaluation**. The API also exposes `/health`,
`/api/evaluations` (POST to run and save; GET to list recent runs), and
interactive API documentation at `/docs`. It supports only the existing
OpenAI-compatible provider configuration and mini models; it does not yet
provide an AI judge, automated quality score, token count, or cost estimate.

The default database is `~/.evalsphere/evalsphere.db`. Set
`EVALSPHERE_DATABASE_URL` to use another SQLite database URL, and
`EVALSPHERE_API_URL` in the Streamlit process to use another API URL. Both
processes should use the same `OPENAI_BASE_URL`. You can configure
`OPENAI_API_KEY` in the API process, or enter it in the Streamlit sidebar; in
the latter case it is sent in a request header to the local API and is not
stored in the database.

The API is intended for local development and has no user authentication or
authorization yet. Keep it bound to loopback (`127.0.0.1`) and do not expose
it to a network until authentication and deployment hardening are added.
Prompts and model responses are persisted locally, so avoid sending sensitive
content unless the local storage and retention are appropriate.

## Models, costs, and quality

The prompt studio offers mini-class chat models only: `gpt-4.1-mini` and
`gpt-4o-mini`. The app uses one chat completion for a generated result after
you review the locally constructed prompt. Image requests include the image in
that completion. Audio requests first use the provider's transcription endpoint
and then send the transcript to the selected mini model.

Mini models can reduce usage costs, but model prices vary by provider and
change over time. Lower-cost models cannot be guaranteed to match premium-model
quality. Review generated locators and code, validate tests in your own
environment, and treat image/audio findings as assistance rather than proof.
The app does not claim that generated XPath selectors have been run against a
live page.

## Configure an API provider

Generation is optional; the prompt builder and local ML benchmark work without
an API key. Set the key as an environment variable or enter it in the app's
sidebar for the current session:

```powershell
$env:OPENAI_API_KEY = "your-provider-key"
```

By default, the app uses `https://api.openai.com/v1`. To use another
OpenAI-compatible endpoint, set `OPENAI_BASE_URL` before starting Streamlit:

```powershell
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"
```

The endpoint must support chat completions and, for audio analysis, the
`/audio/transcriptions` API using the `whisper-1` model name. A local
OpenAI-compatible service may be configured with a loopback HTTP URL. Provider
usage fees may apply to generation, image input, and transcription.

Uploaded images and audio are sent to the configured provider only when you
select **Run with mini model**. The tabular benchmark does not call the
provider. Benchmark results are held in Streamlit session state for the active
session; do not use a shared deployment for confidential data unless its
hosting and retention policies are appropriate.

## Run locally

Use Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

To run the focused provider and prompt-builder tests:

```powershell
python -m unittest discover -s tests
```
#
