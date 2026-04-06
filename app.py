"""Gemma 4 Code Review Agent — Streamlit UI."""

import time

import streamlit as st
from reviewer import (
    review_repo, RepoReview, FileReview, section_is_clean,
    get_available_models, speed_indicator,
)

st.set_page_config(page_title="Gemma 4 Code Reviewer", page_icon="🔍", layout="wide")

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🔍 Gemma 4 Code Review Agent")
st.caption("Local AI-powered code reviews — your code never leaves your machine.")


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}" if m else f"{s}s"


# ── Sidebar: input ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Repository")
    source = st.text_input(
        "GitHub URL or local path",
        placeholder="https://github.com/user/repo or /path/to/project",
    )

    # Model selector with speed indicators
    st.subheader("Model")
    models = get_available_models()
    if not models:
        st.warning("No Ollama models found. Run `ollama pull <model>` first.")
        st.stop()

    model_options = [m["name"] for m in models]
    model_labels = [
        f"{m['name']}  {speed_indicator(m['size_gb'])}  ({m['size_gb']:.1f} GB)"
        for m in models
    ]
    # Default to gemma4 if available, else first model
    default_idx = 0
    for i, name in enumerate(model_options):
        if "gemma4" in name:
            default_idx = i
            break

    selected_model = st.selectbox(
        "Select model",
        options=model_options,
        index=default_idx,
        format_func=lambda x: next(
            lbl for opt, lbl in zip(model_options, model_labels) if opt == x
        ),
    )

    max_files = st.slider("Max files to review", 1, 50, 10)
    run = st.button("Run Review", type="primary", use_container_width=True)

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Clones the repo (or reads local path)\n"
        "2. Extracts reviewable source files\n"
        f"3. Sends each file to **{selected_model}** via Ollama\n"
        "4. Parses structured feedback into categories"
    )

# ── Run review ──────────────────────────────────────────────────────────────
if run and source:
    # Timer and progress at the top of the main area
    timer_col, status_col = st.columns([1, 3])
    with timer_col:
        timer_placeholder = st.empty()
    with status_col:
        status_text = st.empty()
    progress_bar = st.progress(0)
    timing_table = st.empty()

    start_time = time.time()
    file_timings: list[dict] = []

    def on_progress(current, total, filename):
        elapsed = time.time() - start_time
        timer_placeholder.metric("⏱ Elapsed", _format_elapsed(elapsed))
        if current == -1:
            status_text.markdown("**Cloning repository...**")
            progress_bar.progress(0)
        else:
            pct = current / total
            progress_bar.progress(pct)
            status_text.markdown(
                f"**Reviewing file {current + 1} of {total}:** `{filename}`"
            )

    try:
        result: RepoReview = review_repo(
            source, max_files=max_files, model=selected_model, on_progress=on_progress,
        )
    except ValueError as e:
        progress_bar.empty()
        status_text.empty()
        timer_placeholder.empty()
        st.error(f"**Invalid input:** {e}")
        st.stop()
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        timer_placeholder.empty()
        st.error(f"**Review failed:** {e}")
        st.stop()

    total_elapsed = time.time() - start_time
    timer_placeholder.metric("⏱ Total", _format_elapsed(total_elapsed))
    progress_bar.progress(1.0)
    status_text.markdown("**Review complete!**")

    # Build per-file timing list
    for fr in result.files_reviewed:
        file_timings.append({
            "File": fr.filename,
            "Time": _format_elapsed(fr.elapsed),
        })
    file_timings.append({"File": "**Total**", "Time": f"**{_format_elapsed(total_elapsed)}**"})

    st.session_state["result"] = result
    st.session_state["elapsed"] = total_elapsed
    st.session_state["file_timings"] = file_timings
    st.session_state["model_used"] = selected_model

if "result" not in st.session_state:
    st.info("Enter a GitHub URL or local path in the sidebar and click **Run Review**.")
    st.stop()

result: RepoReview = st.session_state["result"]

# ── Timing summary at top ───────────────────────────────────────────────────
if "file_timings" in st.session_state:
    with st.expander(
        f"⏱ Scan times — **{_format_elapsed(st.session_state['elapsed'])}** total"
        f"  ({st.session_state.get('model_used', '')})",
        expanded=False,
    ):
        st.markdown(
            "| File | Time |\n|---|---|\n"
            + "\n".join(
                f"| {t['File']} | {t['Time']} |"
                for t in st.session_state["file_timings"]
            )
        )

# ── Dashboard metrics ───────────────────────────────────────────────────────
reviewed = result.files_reviewed
scores = []
security_count = 0
bug_count = 0

for fr in reviewed:
    # extract numeric score
    for word in fr.score.split():
        try:
            s = float(word.split("/")[0])
            if 1 <= s <= 10:
                scores.append(s)
                break
        except ValueError:
            continue
    if not section_is_clean(fr.security):
        security_count += 1
    if not section_is_clean(fr.bugs):
        bug_count += 1

avg_score = sum(scores) / len(scores) if scores else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Files Reviewed", len(reviewed))
col2.metric("Avg Score", f"{avg_score:.1f}/10")
col3.metric("Security Flags", security_count, delta_color="inverse")
col4.metric("Bug Flags", bug_count, delta_color="inverse")

st.divider()

# ── Category tabs ───────────────────────────────────────────────────────────
tab_overview, tab_security, tab_bugs, tab_style, tab_perf, tab_raw = st.tabs(
    ["Overview", "Security", "Bugs & Logic", "Style", "Performance", "Raw Output"]
)


def _file_icon(fr: FileReview) -> str:
    if fr.error:
        return "❌"
    if not section_is_clean(fr.security):
        return "🔴"
    if not section_is_clean(fr.bugs):
        return "🟡"
    return "🟢"


# ── Overview tab ────────────────────────────────────────────────────────────
with tab_overview:
    for fr in reviewed:
        icon = _file_icon(fr)
        with st.expander(f"{icon}  **{fr.filename}** ({fr.language}) — {fr.score or 'N/A'}"):
            if fr.error:
                st.error(fr.error)
            else:
                st.markdown(fr.summary or "*No summary available.*")

    if result.files_skipped:
        with st.expander(f"Skipped files ({len(result.files_skipped)})"):
            st.code("\n".join(result.files_skipped[:100]))


# ── Section helper ──────────────────────────────────────────────────────────
def _render_section(tab, attr: str, empty_msg: str):
    with tab:
        found_any = False
        for fr in reviewed:
            text = getattr(fr, attr, "")
            if not section_is_clean(text):
                found_any = True
                st.markdown(f"### `{fr.filename}`")
                st.markdown(text)
                st.divider()
        if not found_any:
            st.success(empty_msg)


_render_section(tab_security, "security", "No security issues detected across all files.")
_render_section(tab_bugs, "bugs", "No bugs or logic errors detected across all files.")
_render_section(tab_style, "style", "No style issues detected across all files.")
_render_section(tab_perf, "performance", "No performance concerns detected across all files.")

# ── Raw output tab ──────────────────────────────────────────────────────────
with tab_raw:
    for fr in reviewed:
        with st.expander(f"**{fr.filename}**"):
            if fr.error:
                st.error(fr.error)
            else:
                st.markdown(fr.raw)
