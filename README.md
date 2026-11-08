# EvalSphere

EvalSphere is a central AI and machine-learning workspace for software quality
engineering. It brings prompt improvement, test-automation asset generation,
multimodal evidence analysis, and tabular model benchmarking into one Streamlit
app.

## Workflows

- **Prompt studio:** Build and edit structured prompts for XPath locators,
  Robot Framework automation, test cases, and custom QA tasks.
- **Image analysis:** Submit a screenshot with context to identify visible UI
  issues and generate evidence-based verification steps.
- **Audio analysis:** Transcribe a recording, then use the selected chat model
  to summarize evidence, defects, decisions, or action items.
- **ML benchmark:** Compare logistic regression, decision tree, K-nearest
  neighbors, Gaussian Naive Bayes, and support vector machine classifiers on a
  labeled CSV. This workflow trains on an 80/20 stratified split in the app
  process and does not send the CSV to the generative model.

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
