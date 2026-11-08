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
from benchmark import compare_classifiers


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
        "evidence, and compare lightweight ML classifiers in one workspace."
    )

    metrics = st.columns(4)
    metrics[0].metric("QA workflows", "4", help="XPath, Robot Framework, test cases, and custom prompts")
    metrics[1].metric("Media workflows", "2", help="Image inspection and audio transcription/analysis")
    metrics[2].metric("Mini models", str(len(MINI_MODELS)), help="Only mini-class chat models are offered")
    metrics[3].metric("Local ML models", "5", help="Classifiers trained on your uploaded CSV")

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
            "Compare five classical classifiers on a labeled CSV. Uploaded benchmark "
            "data stays in this app session and is not sent to the chat model."
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


def _read_uploaded_csv(raw_data: bytes) -> pd.DataFrame:
    try:
        data = pd.read_csv(BytesIO(raw_data))
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as error:
        raise ValueError(f"Could not read the CSV: {error}") from error
    if data.empty:
        raise ValueError("The uploaded CSV has no data rows.")
    if len(data.columns) == 0:
        raise ValueError("The CSV must have a header row with feature and target columns.")
    return data


def render_ml_benchmark() -> None:
    st.title("ML benchmark", icon=":material/model_training:")
    st.write(
        "Compare five classical classifiers on a labeled CSV. Training and evaluation "
        "run in this app process; the dataset is not sent to the generative model."
    )
    uploaded_file = st.file_uploader(
        "Upload a labeled CSV",
        type=("csv",),
        max_upload_size=20,
        key="benchmark_csv",
    )
    if uploaded_file is None:
        st.info("Upload a CSV with at least 20 labeled rows and one feature column.")
        return

    raw_data = uploaded_file.getvalue()
    if len(raw_data) > MAX_UPLOAD_BYTES:
        st.error("The CSV exceeds the 20 MB upload limit.")
        return
    try:
        data = _read_uploaded_csv(raw_data)
    except ValueError as error:
        st.error(str(error))
        return
    if data.columns.duplicated().any():
        st.error("CSV column names must be unique.")
        return

    target_column = st.selectbox(
        "Target column",
        list(data.columns),
        index=len(data.columns) - 1,
        key="benchmark_target",
    )
    st.caption(f"{len(data):,} rows · {len(data.columns) - 1:,} candidate features")
    run_clicked = st.button("Compare classifiers", type="primary", key="run_benchmark")
    input_fingerprint = hashlib.sha256(raw_data).hexdigest()

    if run_clicked:
        try:
            with st.spinner("Training and comparing five classifiers..."):
                result = compare_classifiers(data, target_column)
                st.session_state["benchmark_result"] = {
                    "fingerprint": input_fingerprint,
                    "target": target_column,
                    "result": result,
                }
        except ValueError as error:
            st.error(str(error))

    saved = st.session_state.get("benchmark_result")
    if not saved:
        return
    if saved["fingerprint"] != input_fingerprint or saved["target"] != target_column:
        st.info("The CSV or target changed. Run the comparison again to refresh the results.")
        return

    results, predictions, y_test, class_labels, feature_importance = saved["result"]
    winner = results.sort_values("F1", ascending=False).iloc[0]
    best_col, samples_col, feature_col = st.columns(3)
    best_col.metric("Top model by weighted F1", winner["Model"], f'{winner["F1"]:.3f} F1')
    samples_col.metric("Holdout rows", f"{len(y_test):,}")
    feature_col.metric("Features", f"{data.shape[1] - 1:,}")

    st.subheader("Model comparison")
    st.dataframe(
        results.round(4),
        hide_index=True,
        width="stretch",
        alt="Accuracy, AUC, precision, recall, F1, MCC, and timing for five classifiers",
    )
    chart_data = results.set_index("Model")[["Accuracy", "F1", "MCC"]]
    st.bar_chart(
        chart_data,
        y_label="Score",
        height=300,
        alt="Accuracy, F1, and Matthews correlation by classifier",
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
