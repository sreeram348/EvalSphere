"""EvalSphere: a focused AI and ML workspace for software quality engineering."""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ai_hub import (
    AIServiceError,
    MINI_MODELS,
    ProviderConfig,
    build_optimized_prompt,
    complete_chat,
    transcribe_audio,
)
from benchmark import (
    SUPPORTED_EXTENSIONS,
    compare_classifiers,
    compare_clusters,
    compare_regressors,
    dataset_sources,
    read_dataset,
)


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
TASKS = (
    "XPath locator",
    "Robot Framework code",
    "Test cases",
    "Image analysis",
    "Audio analysis",
    "Custom QA prompt",
)


def render_sidebar() -> tuple[str, ProviderConfig]:
    st.sidebar.title("EvalSphere")
    st.sidebar.caption("AI workspace for software quality engineering")
    page = st.sidebar.radio(
        "Workspace",
        ("Overview", "Prompt studio", "ML benchmark"),
        key="workspace_page",
    )
    if page is None:
        raise RuntimeError("No workspace page was selected.")
    st.sidebar.subheader("Mini model settings")
    api_key = st.sidebar.text_input(
        "API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Used only for the current session. Set OPENAI_API_KEY in the environment instead.",
        key="provider_api_key",
    )
    model = st.sidebar.selectbox(
        "Model",
        MINI_MODELS,
        index=0,
        key="provider_model",
    )
    if model is None:
        raise RuntimeError("No provider model was selected.")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    st.sidebar.caption(f"API endpoint: `{base_url}`")
    st.sidebar.caption(
        "Mini models keep usage costs lower. Pricing and quality depend on your provider; "
        "premium-level results are not guaranteed."
    )
    return page, ProviderConfig(api_key=api_key.strip(), base_url=base_url, model=model)


def render_overview() -> None:
    st.title("Your AI quality engineering hub", icon=":material/hub:")
    st.write(
        "Build stronger prompts, generate automation assets, analyze visual and audio "
        "evidence, and compare classification, regression, and clustering models."
    )

    metrics = st.columns(4)
    metrics[0].metric("QA workflows", "4", help="XPath, Robot Framework, test cases, and custom prompts")
    metrics[1].metric("Media workflows", "2", help="Image inspection and audio transcription/analysis")
    metrics[2].metric("Mini models", str(len(MINI_MODELS)), help="Only mini-class chat models are offered")
    metrics[3].metric(
        "Local ML models",
        "9 + 8",
        help="Nine classifiers and eight regressors, plus unsupervised clustering",
    )

    st.subheader("Choose a workflow")
    cards = st.columns(3)
    with cards[0].container(border=True):
        st.markdown("#### :material/psychology: Prompt studio")
        st.write(
            "Turn a short request into a structured, editable prompt, then generate "
            "XPath locators, Robot Framework code, or test cases."
        )
    with cards[1].container(border=True):
        st.markdown("#### :material/image_search: Evidence analysis")
        st.write(
            "Inspect screenshots for visible UI issues and transcribe recordings before "
            "summarizing defects, decisions, or action items."
        )
    with cards[2].container(border=True):
        st.markdown("#### :material/model_training: ML benchmark")
        st.write(
            "Compare classification, regression, and clustering models on local tabular "
            "data. Uploaded benchmark data stays in this app session."
        )

    st.info(
        "AI generation requires an API key from an OpenAI-compatible provider. "
        "Configure `OPENAI_API_KEY` and, when needed, `OPENAI_BASE_URL`; provider "
        "charges may apply. Audio transcription uses the provider's transcription endpoint.",
        title="Before you start",
    )


def _tool_controls(task: str) -> tuple[str, str, str, Any]:
    task_key = task.lower().replace(" ", "_")
    if task == "XPath locator":
        request_label = "What should the locator find?"
        context_label = "DOM or HTML snippet"
        context_help = "Paste the relevant inspected markup. Locators cannot be verified without the page DOM."
        output_format = st.selectbox(
            "Output format",
            ("XPath + CSS fallback", "XPath only", "JSON"),
            key=f"format_{task_key}",
        )
        framework = "Browser-neutral"
        attachment = None
    elif task == "Robot Framework code":
        request_label = "Describe the user flow or behavior to automate"
        context_label = "Application details, selectors, or existing code"
        context_help = "Add the page structure, base URL, known locators, and relevant libraries."
        framework = st.selectbox(
            "Automation library",
            ("Browser (Playwright)", "SeleniumLibrary"),
            key=f"framework_{task_key}",
        )
        output_format = "Executable Robot Framework code with assumptions"
        attachment = None
    elif task == "Test cases":
        request_label = "Describe the feature, story, or change"
        context_label = "Acceptance criteria and constraints"
        context_help = "Include roles, edge cases, supported platforms, and any known limitations."
        framework = "Framework-neutral"
        output_format = st.selectbox(
            "Output format",
            ("Markdown test matrix", "Gherkin scenarios", "JSON"),
            key=f"format_{task_key}",
        )
        attachment = None
    elif task == "Image analysis":
        request_label = "What should the image analysis focus on?"
        context_label = "Screen, product, or test context"
        context_help = "Explain expected behavior or the UI area under test."
        framework = "Visual evidence"
        output_format = "Prioritized findings with visual evidence and verification steps"
        attachment = st.file_uploader(
            "Upload an image",
            type=("png", "jpg", "jpeg", "webp"),
            max_upload_size=20,
            key=f"media_{task_key}",
        )
    elif task == "Audio analysis":
        request_label = "What should the recording analysis focus on?"
        context_label = "Meeting, call, or test-run context"
        context_help = "For example: extract defects, decisions, speaker concerns, or action items."
        framework = "Audio transcription and analysis"
        output_format = "Transcript summary, evidence, and prioritized findings"
        attachment = st.file_uploader(
            "Upload an audio recording",
            type=("wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm", "ogg", "flac"),
            max_upload_size=20,
            key=f"media_{task_key}",
        )
    else:
        request_label = "What do you need the model to do?"
        context_label = "Relevant context or source material"
        context_help = "Include constraints, audience, and what a good result should contain."
        framework = "User-defined"
        output_format = st.selectbox(
            "Output format",
            ("Concise checklist", "Markdown", "JSON", "Code with explanation"),
            key=f"format_{task_key}",
        )
        attachment = None

    return request_label, context_label, context_help, (framework, output_format, attachment)


def render_prompt_studio(config: ProviderConfig) -> None:
    st.title("Prompt studio", icon=":material/auto_awesome:")
    st.write(
        "Start with the task and evidence. EvalSphere builds a detailed prompt locally; "
        "review or edit it before sending anything to your mini model."
    )
    task = st.selectbox("Workflow", TASKS, key="prompt_task")
    if task is None:
        raise RuntimeError("No prompt workflow was selected.")
    request_label, context_label, context_help, options = _tool_controls(task)
    framework, output_format, attachment = options
    task_key = task.lower().replace(" ", "_")
    editor_key = f"optimized_prompt_{task_key}"
    media_key = f"media_payload_{task_key}"

    with st.form(f"prompt_form_{task_key}"):
        request_text = st.text_area(
            request_label,
            placeholder="Describe the expected outcome, behavior, or question.",
            height=110,
            key=f"request_{task_key}",
        )
        context_text = st.text_area(
            context_label,
            help=context_help,
            height=160,
            key=f"context_{task_key}",
        )
        submitted = st.form_submit_button(
            "Build optimized prompt",
            type="primary",
            key=f"build_{task_key}",
        )

    if submitted:
        st.session_state.pop(editor_key, None)
        st.session_state.pop(media_key, None)
        st.session_state.pop(f"output_{task_key}", None)
        st.session_state.pop(f"transcript_{task_key}", None)
        if not request_text.strip():
            st.error("Describe the task before building a prompt.")
        elif task in ("Image analysis", "Audio analysis") and attachment is None:
            st.error("Upload an image or audio file for this workflow.")
        elif attachment is not None and len(attachment.getvalue()) > MAX_UPLOAD_BYTES:
            st.error("The media file exceeds the 20 MB limit.")
        else:
            prompt = build_optimized_prompt(
                task=task,
                request=request_text,
                context=context_text,
                framework=framework,
                output_format=output_format,
            )
            st.session_state[editor_key] = prompt
            if attachment is not None:
                content = attachment.getvalue()
                st.session_state[media_key] = {
                    "name": Path(attachment.name).name,
                    "mime_type": attachment.type or "application/octet-stream",
                    "content": content,
                    "fingerprint": hashlib.sha256(content).hexdigest(),
                }
            else:
                st.session_state.pop(media_key, None)
    elif task in ("Image analysis", "Audio analysis"):
        stored_media = st.session_state.get(media_key)
        uploaded_content = attachment.getvalue() if attachment is not None else None
        uploaded_fingerprint = (
            hashlib.sha256(uploaded_content).hexdigest()
            if uploaded_content is not None
            else None
        )
        if stored_media and uploaded_fingerprint != stored_media["fingerprint"]:
            st.session_state.pop(editor_key, None)
            st.session_state.pop(media_key, None)
            st.session_state.pop(f"output_{task_key}", None)
            st.session_state.pop(f"transcript_{task_key}", None)
            st.info("The media file changed. Build the prompt again to use the new evidence.")

    if editor_key not in st.session_state:
        st.caption("Build a prompt to review and edit it before generation.")
        return

    st.subheader("Review your optimized prompt")
    editable_prompt = st.text_area(
        "Optimized prompt (editable)",
        key=editor_key,
        height=300,
    )
    media = st.session_state.get(media_key)
    if media and task == "Image analysis":
        st.image(media["content"], alt="Uploaded UI evidence for analysis", width="stretch")
    elif media and task == "Audio analysis":
        st.audio(media["content"], format=media["mime_type"], alt="Uploaded audio for analysis")

    if st.button("Run with mini model", type="primary", key=f"run_{task_key}"):
        st.session_state.pop(f"output_{task_key}", None)
        st.session_state.pop(f"transcript_{task_key}", None)
        if not config.api_key:
            st.error("Set `OPENAI_API_KEY` or enter an API key in the sidebar before generating.")
        elif not editable_prompt.strip():
            st.error("The prompt cannot be empty.")
        elif task in ("Image analysis", "Audio analysis") and not media:
            st.error("Build the prompt again and attach the required media file.")
        else:
            try:
                with st.spinner("Generating with the selected mini model..."):
                    if task == "Audio analysis":
                        transcript = transcribe_audio(
                            config=config,
                            audio=media["content"],
                            filename=media["name"],
                        )
                        st.session_state[f"transcript_{task_key}"] = transcript
                        response = complete_chat(
                            config=config,
                            prompt=(
                                f"{editable_prompt}\n\n"
                                "Analyze this transcript. Clearly separate directly stated "
                                "evidence from interpretation:\n\n"
                                f"{transcript}"
                            ),
                        )
                    else:
                        image = media if task == "Image analysis" else None
                        response = complete_chat(
                            config=config,
                            prompt=editable_prompt,
                            image=image,
                        )
                    st.session_state[f"output_{task_key}"] = response
            except AIServiceError as error:
                st.error(str(error), icon=":material/error:")

    transcript = st.session_state.get(f"transcript_{task_key}")
    if transcript:
        with st.expander("Audio transcript"):
            st.text(transcript)

    output = st.session_state.get(f"output_{task_key}")
    if output:
        st.subheader("Generated result")
        st.code(output, language=None)
        safe_task_name = task_key.replace("_", "-")
        st.download_button(
            "Download result",
            data=output,
            file_name=f"evalsphere-{safe_task_name}.txt",
            mime="text/plain",
            key=f"download_{task_key}",
        )


def _suggest_target_column(
    data: pd.DataFrame,
    analysis_type: str,
) -> str:
    columns = list(data.columns)
    if not columns:
        raise ValueError("The uploaded dataset has no columns.")

    label_names = {"target", "label", "class", "outcome", "y"}
    named_targets = [
        column for column in columns if str(column).strip().lower() in label_names
    ]
    if analysis_type == "Regression (supervised)":
        named_numeric_targets = [
            column
            for column in named_targets
            if pd.to_numeric(data[column], errors="coerce").notna().all()
            and data[column].nunique(dropna=True) > 2
        ]
        if named_numeric_targets:
            return named_numeric_targets[-1]
        numeric_targets = [
            column
            for column in columns
            if pd.to_numeric(data[column], errors="coerce").notna().all()
            and data[column].nunique(dropna=True) > 2
        ]
        if numeric_targets:
            return numeric_targets[-1]
    else:
        named_class_targets = [
            column
            for column in named_targets
            if 2 <= data[column].nunique(dropna=True) <= 50
        ]
        if named_class_targets:
            return named_class_targets[-1]
        class_limit = min(50, max(2, len(data) // 10))
        low_cardinality = [
            column
            for column in columns
            if 2 <= data[column].nunique(dropna=True) <= class_limit
        ]
        categorical_targets = [
            column
            for column in low_cardinality
            if not pd.api.types.is_numeric_dtype(data[column])
        ]
        if categorical_targets:
            return categorical_targets[-1]
        if low_cardinality:
            return low_cardinality[-1]
    return columns[-1]


def render_ml_benchmark() -> None:
    st.title("ML benchmark", icon=":material/model_training:")
    st.write(
        "Benchmark supervised classification and regression, or unsupervised clustering. "
        "Training and evaluation run locally in this app; uploaded data is not sent to "
        "the generative model."
    )
    uploaded_file = st.file_uploader(
        "Upload a tabular dataset",
        type=tuple(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS),
        max_upload_size=20,
        key="benchmark_file",
    )
    if uploaded_file is None:
        st.info(
            "Supported formats: CSV, Excel (.xlsx), JSON, Parquet, HTML tables, "
            "XML tables, and Markdown pipe tables. Maximum upload size: 20 MB."
        )
        return

    raw_data = uploaded_file.getvalue()
    if len(raw_data) > MAX_UPLOAD_BYTES:
        st.error("The uploaded file exceeds the 20 MB upload limit.")
        return
    try:
        sources = dataset_sources(uploaded_file.name, raw_data)
        source = None
        if len(sources) > 1:
            source = st.selectbox(
                "Worksheet or HTML table",
                sources,
                key="benchmark_source",
            )
        data = read_dataset(uploaded_file.name, raw_data, source=source)
    except ValueError as error:
        st.error(str(error))
        return
    if data.columns.duplicated().any():
        st.error("Dataset column names must be unique.")
        return
    if any(not isinstance(column, str) or not column.strip() for column in data.columns):
        st.error("Dataset columns must have non-empty text names.")
        return

    st.caption(
        f"{len(data):,} rows · {len(data.columns):,} columns · {uploaded_file.name}"
    )
    with st.expander("Preview uploaded data"):
        st.dataframe(
            data.head(10),
            hide_index=True,
            width="stretch",
            alt="First ten rows of the uploaded dataset",
        )

    analysis_type = st.selectbox(
        "Analysis type",
        (
            "Classification (supervised)",
            "Regression (supervised)",
            "Clustering (unsupervised)",
        ),
        key="benchmark_type",
    )
    if analysis_type is None:
        raise RuntimeError("No benchmark analysis type was selected.")
    target_column = None
    excluded_columns: list[str] = []
    n_clusters = 3
    tune_best = False
    if analysis_type != "Clustering (unsupervised)":
        target_options = list(data.columns)
        suggested_target = _suggest_target_column(data, analysis_type)
        target_column = st.selectbox(
            "Target column",
            target_options,
            index=target_options.index(suggested_target),
            key=(
                "benchmark_target_classification"
                if analysis_type == "Classification (supervised)"
                else "benchmark_target_regression"
            ),
        )
        st.caption(f"Suggested target: `{suggested_target}`. Review before running.")
        if analysis_type in {
            "Classification (supervised)",
            "Regression (supervised)",
        }:
            tune_best = st.checkbox(
                "Automatically tune the leading model",
                value=True,
                key="benchmark_tune",
                help=(
                    "Runs a bounded cross-validation search on the leading model's "
                    "training partition. The held-out ranking remains dataset-specific."
                ),
            )
    else:
        n_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=min(10, max(2, len(data) - 1)),
            value=min(3, max(2, len(data) - 1)),
            key="benchmark_cluster_count",
        )
        excluded_columns = st.multiselect(
            "Columns to exclude",
            list(data.columns),
            help="Exclude identifiers or known labels so they do not dominate clusters.",
            key="benchmark_cluster_exclusions",
        )

    fingerprint_parts = [
        hashlib.sha256(raw_data).hexdigest(),
        uploaded_file.name,
        str(source),
        analysis_type,
        str(target_column),
        str(n_clusters),
        repr(sorted(excluded_columns)),
        str(tune_best),
    ]
    input_fingerprint = hashlib.sha256("|".join(fingerprint_parts).encode()).hexdigest()
    run_clicked = st.button(
        "Run benchmark",
        type="primary",
        key="run_benchmark",
    )

    if run_clicked:
        try:
            with st.spinner("Preparing data and benchmarking models..."):
                if analysis_type == "Classification (supervised)":
                    if target_column is None:
                        raise ValueError("Choose a target column.")
                    result = compare_classifiers(
                        data, target_column, tune_best=tune_best
                    )
                elif analysis_type == "Regression (supervised)":
                    if target_column is None:
                        raise ValueError("Choose a target column.")
                    result = compare_regressors(
                        data, target_column, tune_best=tune_best
                    )
                else:
                    result = compare_clusters(
                        data,
                        excluded_columns=tuple(excluded_columns),
                        n_clusters=n_clusters,
                    )
                st.session_state["benchmark_result"] = {
                    "fingerprint": input_fingerprint,
                    "analysis_type": analysis_type,
                    "result": result,
                }
        except ValueError as error:
            st.error(str(error))

    saved = st.session_state.get("benchmark_result")
    if not saved:
        return
    if (
        saved["fingerprint"] != input_fingerprint
        or saved["analysis_type"] != analysis_type
    ):
        st.info("The dataset or benchmark settings changed. Run the benchmark again.")
        return

    st.subheader("Model comparison")
    if analysis_type == "Classification (supervised)":
        results, predictions, y_test, class_labels, feature_importance = saved["result"]
        winner = results.iloc[0]
        best_col, samples_col, feature_col = st.columns(3)
        best_col.metric(
            "Top model by weighted F1",
            winner["Model"],
            f'{winner["F1"]:.3f} F1',
        )
        samples_col.metric("Holdout rows", f"{len(y_test):,}")
        feature_col.metric("Features", f"{data.shape[1] - 1:,}")
        st.dataframe(
            results.round(4),
            hide_index=True,
            width="stretch",
            alt="Ranked classification models with quality and runtime metrics",
        )
        st.bar_chart(
            results.set_index("Model")[["F1", "AUC", "MCC"]],
            y_label="Score",
            height=300,
            alt="Classification quality scores, ranked by weighted F1",
        )
        selected_model = st.selectbox(
            "Inspect a model",
            results["Model"].tolist(),
            key="inspect_model",
        )
        actual = y_test.reset_index(drop=True)
        predicted = pd.Series(predictions[selected_model], name="Predicted")
        report = pd.DataFrame(
            classification_report(
                actual,
                predicted,
                labels=class_labels,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()
        confusion = pd.DataFrame(
            confusion_matrix(actual, predicted, labels=class_labels),
            index=pd.Index(class_labels, name="Actual"),
            columns=pd.Index(class_labels, name="Predicted"),
        )
        report_col, matrix_col = st.columns(2)
        with report_col:
            st.subheader("Classification report")
            st.dataframe(
                report.round(3),
                width="stretch",
                alt=f"Per-class classification metrics for {selected_model}",
            )
        with matrix_col:
            st.subheader("Confusion matrix")
            st.dataframe(
                confusion,
                width="stretch",
                alt=f"Actual versus predicted classes for {selected_model}",
            )
        st.subheader("Feature relevance")
        st.caption("Mutual information estimated from the training partition.")
        st.dataframe(
            feature_importance.head(15).round(4),
            hide_index=True,
            width="stretch",
            alt="Top fifteen features by training-partition mutual information",
        )
        st.caption(
            "The top rank is specific to this dataset and split, not a universal model ranking."
        )
    elif analysis_type == "Regression (supervised)":
        results, predictions, y_test = saved["result"]
        winner = results.iloc[0]
        best_col, samples_col = st.columns(2)
        best_col.metric("Top model by R²", winner["Model"], f'{winner["R2"]:.3f} R²')
        samples_col.metric("Holdout rows", f"{len(y_test):,}")
        st.dataframe(
            results.round(4),
            hide_index=True,
            width="stretch",
            alt="Ranked regression models with R-squared, error, and runtime metrics",
        )
        st.bar_chart(
            results.set_index("Model")[["R2"]],
            y_label="R² score",
            height=300,
            alt="Regression model R-squared scores, ranked highest first",
        )
        selected_model = st.selectbox(
            "Inspect a model",
            results["Model"].tolist(),
            key="inspect_regressor",
        )
        prediction_preview = pd.DataFrame(
            {
                "Actual": y_test.reset_index(drop=True),
                "Predicted": predictions[selected_model],
            }
        )
        st.dataframe(
            prediction_preview.head(100),
            hide_index=True,
            width="stretch",
            alt=f"Actual and predicted target values for {selected_model}",
        )
        st.caption("The top rank is specific to this dataset and holdout split.")
    else:
        results, labels_by_model, _ = saved["result"]
        winner = results.iloc[0]
        st.metric(
            "Top clustering model by silhouette",
            winner["Model"],
            f'{winner["Silhouette"]:.3f} silhouette'
            if pd.notna(winner["Silhouette"])
            else "Not defined",
        )
        st.dataframe(
            results.round(4),
            hide_index=True,
            width="stretch",
            alt="Ranked clustering models with sampled cluster-quality metrics",
        )
        st.bar_chart(
            results.set_index("Model")[["Silhouette"]],
            y_label="Silhouette score",
            height=300,
            alt="Clustering models ranked by sampled silhouette score",
        )
        selected_model = st.selectbox(
            "Inspect cluster assignments",
            results["Model"].tolist(),
            key="inspect_cluster_model",
        )
        assignments = pd.DataFrame(
            {"Cluster": labels_by_model[selected_model]}, index=data.index
        )
        st.dataframe(
            assignments.head(500),
            width="stretch",
            alt=f"First 500 cluster assignments from {selected_model}",
        )
        st.caption(
            "Clustering quality scores are computed on a deterministic sample of up to "
            f"{2_000:,} rows; clusters are not ground-truth labels."
        )


st.set_page_config(
    page_title="EvalSphere | AI quality hub",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_name, provider_config = render_sidebar()
if page_name == "Overview":
    render_overview()
elif page_name == "Prompt studio":
    render_prompt_studio(provider_config)
else:
    render_ml_benchmark()
