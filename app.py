"""Gemma 4 Code Review Agent — Streamlit UI."""

import re
import time
from collections import defaultdict
from datetime import datetime

import streamlit as st
from reviewer import (
    review_repo, RepoReview, FileReview, section_is_clean,
    get_available_models, speed_indicator,
)

st.set_page_config(page_title="Gemma 4 Code Reviewer", page_icon="🔍", layout="wide")

# ── Compact styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    [data-testid="stMetric"] { padding: 0.3rem 0; }
    [data-testid="stExpander"] { margin-bottom: 0.25rem; }
    h1 { margin-bottom: 0 !important; font-size: 1.8rem !important; }
    h3 { margin-top: 0.5rem !important; margin-bottom: 0.25rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stProgress > div { margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🔍 Gemma 4 Code Review Agent")
st.caption("Local AI-powered code reviews — your code never leaves your machine.")


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}" if m else f"{s}s"


# Score regex: matches "7/10", "7 / 10", "7 out of 10", or a standalone
# integer 1-10 near the word "score" context (already in the score section).
_SCORE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*10"  # "7/10" or "7 out of 10"
    r"|"
    r"\b(\d+(?:\.\d+)?)\b"                      # fallback: any number
)


def _extract_score(score_text: str) -> float | None:
    """Extract a numeric score (1-10) from the score section text."""
    for m in _SCORE_RE.finditer(score_text):
        val = m.group(1) or m.group(2)
        try:
            s = float(val)
            if 1 <= s <= 10:
                return s
        except ValueError:
            continue
    return None


# ── Initialize history in session state ─────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = {}

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

# ── Main layout: results (left) + history (right) ──────────────────────────
if "history_width" not in st.session_state:
    st.session_state["history_width"] = 2
history_width = st.session_state["history_width"]
main_col, history_col = st.columns([6 - history_width, history_width])

# ── Run review ──────────────────────────────────────────────────────────────
if run and source:
    with main_col:
        timer_col, status_col = st.columns([1, 3])
        with timer_col:
            timer_placeholder = st.empty()
        with status_col:
            status_text = st.empty()
        progress_bar = st.progress(0)

    start_time = time.time()

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
        with main_col:
            st.error(f"**Invalid input:** {e}")
        st.stop()
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        timer_placeholder.empty()
        with main_col:
            st.error(f"**Review failed:** {e}")
        st.stop()

    total_elapsed = time.time() - start_time
    timer_placeholder.metric("⏱ Total", _format_elapsed(total_elapsed))
    progress_bar.progress(1.0)
    status_text.markdown("**Review complete!**")

    # Build per-file timing list
    file_timings = []
    for fr in result.files_reviewed:
        file_timings.append({"File": fr.filename, "Time": _format_elapsed(fr.elapsed)})
    file_timings.append({"File": "**Total**", "Time": f"**{_format_elapsed(total_elapsed)}**"})

    # Compute avg score for history
    scores_for_hist = []
    for fr in result.files_reviewed:
        s = _extract_score(fr.score)
        if s is not None:
            scores_for_hist.append(s)
    avg_for_hist = sum(scores_for_hist) / len(scores_for_hist) if scores_for_hist else 0

    st.session_state["result"] = result
    st.session_state["elapsed"] = total_elapsed
    st.session_state["file_timings"] = file_timings
    st.session_state["model_used"] = selected_model

    # Build per-file detail breakdown for comparison and consensus
    file_scores = {}
    file_details = {}
    for fr in result.files_reviewed:
        s = _extract_score(fr.score)
        file_scores[fr.filename] = s
        file_details[fr.filename] = {
            "security": fr.security,
            "bugs": fr.bugs,
            "style": fr.style,
            "performance": fr.performance,
            "summary": fr.summary,
        }

    # Update history — keyed by (source, model) so re-scans replace
    display_name = source.rstrip("/").split("/")[-1]
    history_key = f"{source}||{selected_model}"
    st.session_state["history"][history_key] = {
        "source": display_name,
        "full_source": source,
        "model": selected_model,
        "elapsed": total_elapsed,
        "avg_score": avg_for_hist,
        "files": len(result.files_reviewed),
        "security_flags": sum(1 for fr in result.files_reviewed if not section_is_clean(fr.security)),
        "bug_flags": sum(1 for fr in result.files_reviewed if not section_is_clean(fr.bugs)),
        "file_scores": file_scores,
        "file_details": file_details,
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
    }

# ── History panel (right column) ────────────────────────────────────────────
with history_col:
    new_width = st.slider("◀ Panel width ▶", 1, 4, history_width, key="width_slider")
    if new_width != history_width:
        st.session_state["history_width"] = new_width
        st.rerun()

    st.markdown("#### Scan History")
    history = st.session_state["history"]
    if not history:
        st.caption("No scans yet.")
    else:
        entries = list(reversed(list(history.values())))

        header = "| Repo | Model | Score | Time | Files | When |\n|---|---|---|---|---|---|"
        rows = []
        for entry in entries:
            score_icon = "🟢" if entry["avg_score"] >= 7 else "🟡" if entry["avg_score"] >= 5 else "🔴"
            rows.append(
                f"| {entry['source']} | {entry['model']} "
                f"| {score_icon} {entry['avg_score']:.1f} "
                f"| {_format_elapsed(entry['elapsed'])} "
                f"| {entry['files']} | {entry['timestamp']} |"
            )
        st.markdown(header + "\n" + "\n".join(rows))

        # ── Model comparison ────────────────────────────────────────────────
        # Group by repo to find repos scanned with multiple models
        repos = defaultdict(list)
        for entry in entries:
            repos[entry["full_source"]].append(entry)

        # Only show comparison if at least one repo has multiple model runs
        multi_model_repos = {k: v for k, v in repos.items() if len(v) > 1}
        if multi_model_repos:
            st.markdown("#### Model Comparison")
            for full_source, runs in multi_model_repos.items():
                display = runs[0]["source"]
                st.markdown(f"**{display}**")

                # Comparison table header
                comp_header = "| Metric |"
                comp_sep = "|---|"
                for r in runs:
                    comp_header += f" {r['model']} |"
                    comp_sep += "---|"

                # Score row with bar visualization
                score_row = "| Avg Score |"
                for r in runs:
                    icon = "🟢" if r["avg_score"] >= 7 else "🟡" if r["avg_score"] >= 5 else "🔴"
                    bar = "█" * int(r["avg_score"]) + "░" * (10 - int(r["avg_score"]))
                    score_row += f" {icon} **{r['avg_score']:.1f}** `{bar}` |"

                time_row = "| Time |"
                for r in runs:
                    time_row += f" {_format_elapsed(r['elapsed'])} |"

                security_row = "| Security |"
                for r in runs:
                    flags = r.get("security_flags", "–")
                    security_row += f" {flags} |"

                bugs_row = "| Bugs |"
                for r in runs:
                    flags = r.get("bug_flags", "–")
                    bugs_row += f" {flags} |"

                st.markdown("\n".join([comp_header, comp_sep, score_row, time_row, security_row, bugs_row]))

                # Per-file score comparison if available
                all_files = set()
                for r in runs:
                    all_files.update(r.get("file_scores", {}).keys())

                if all_files:
                    with st.expander("Per-file scores"):
                        file_header = "| File |"
                        file_sep = "|---|"
                        for r in runs:
                            file_header += f" {r['model']} |"
                            file_sep += "---|"

                        file_rows = []
                        for fname in sorted(all_files):
                            row = f"| `{fname}` |"
                            for r in runs:
                                s = r.get("file_scores", {}).get(fname)
                                if s is not None:
                                    icon = "🟢" if s >= 7 else "🟡" if s >= 5 else "🔴"
                                    row += f" {icon} {s:.0f} |"
                                else:
                                    row += " – |"
                            file_rows.append(row)
                        st.markdown("\n".join([file_header, file_sep] + file_rows))

                st.divider()

    if history:
        if st.button("Clear History", use_container_width=True):
            st.session_state["history"] = {}
            st.rerun()

# ── Check for results to display ────────────────────────────────────────────
if "result" not in st.session_state:
    with main_col:
        st.info("Enter a GitHub URL or local path in the sidebar and click **Run Review**.")
    st.stop()

result: RepoReview = st.session_state["result"]

# ── Results ─────────────────────────────────────────────────────────────────
with main_col:
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

    # ── Dashboard metrics ───────────────────────────────────────────────────
    reviewed = result.files_reviewed
    scores = []
    security_count = 0
    bug_count = 0

    for fr in reviewed:
        s = _extract_score(fr.score)
        if s is not None:
            scores.append(s)
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

    # ── Category tabs ───────────────────────────────────────────────────────
    # Show Consensus tab when multiple models have scanned the same repo
    current_source = st.session_state.get("result", RepoReview("")).repo_source
    history = st.session_state.get("history", {})
    multi_model_runs = [
        e for e in history.values()
        if e["full_source"] == current_source
    ]
    has_consensus = len(multi_model_runs) > 1

    if has_consensus:
        tab_overview, tab_consensus, tab_security, tab_bugs, tab_style, tab_perf, tab_raw = st.tabs(
            ["Overview", "⚖ Consensus", "Security", "Bugs & Logic", "Style", "Performance", "Raw Output"]
        )
    else:
        tab_overview, tab_security, tab_bugs, tab_style, tab_perf, tab_raw = st.tabs(
            ["Overview", "Security", "Bugs & Logic", "Style", "Performance", "Raw Output"]
        )
        tab_consensus = None

    def _file_icon(fr: FileReview) -> str:
        if fr.error:
            return "❌"
        if not section_is_clean(fr.security):
            return "🔴"
        if not section_is_clean(fr.bugs):
            return "🟡"
        return "🟢"

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

    # ── Consensus tab ─────────────────────────────────────────────────────
    if tab_consensus is not None and has_consensus:
        with tab_consensus:
            st.markdown(
                "Cross-references findings from **all model runs** on this repo. "
                "Issues flagged by multiple models are high-confidence action items."
            )

            model_names = [r["model"] for r in multi_model_runs]

            for category, label in [
                ("security", "🔒 Security"),
                ("bugs", "🐛 Bugs & Logic"),
                ("style", "✏ Style"),
                ("performance", "⚡ Performance"),
            ]:
                # Collect per-file: which models flagged issues
                all_files = set()
                for r in multi_model_runs:
                    all_files.update(r.get("file_details", {}).keys())

                flagged_files = []
                for fname in sorted(all_files):
                    models_flagging = []
                    findings_by_model = {}
                    for r in multi_model_runs:
                        detail = r.get("file_details", {}).get(fname, {})
                        text = detail.get(category, "")
                        if not section_is_clean(text):
                            models_flagging.append(r["model"])
                            findings_by_model[r["model"]] = text
                    if models_flagging:
                        flagged_files.append((fname, models_flagging, findings_by_model))

                if not flagged_files:
                    continue

                st.markdown(f"### {label}")
                for fname, models_flagging, findings in flagged_files:
                    agreement = len(models_flagging)
                    total_models = len(multi_model_runs)

                    if agreement == total_models:
                        conf = "🔴 **All models agree**"
                    elif agreement > 1:
                        conf = f"🟡 **{agreement}/{total_models} models agree**"
                    else:
                        conf = f"⚪ **1 model only** ({models_flagging[0]})"

                    with st.expander(f"`{fname}` — {conf}"):
                        for model_name, text in findings.items():
                            st.markdown(f"**{model_name}:**")
                            st.markdown(text)
                            st.markdown("---")

            # Summary action items
            st.markdown("### 📋 Priority Action Items")
            action_items = []
            all_files = set()
            for r in multi_model_runs:
                all_files.update(r.get("file_details", {}).keys())

            for fname in sorted(all_files):
                for category, emoji in [("security", "🔒"), ("bugs", "🐛")]:
                    count = 0
                    for r in multi_model_runs:
                        detail = r.get("file_details", {}).get(fname, {})
                        text = detail.get(category, "")
                        if not section_is_clean(text):
                            count += 1
                    if count > 1:
                        action_items.append(
                            f"- {emoji} **`{fname}`** — {category} issue confirmed by "
                            f"**{count}/{len(multi_model_runs)}** models"
                        )
                    elif count == 1:
                        action_items.append(
                            f"- {emoji} `{fname}` — {category} flagged by 1 model (review recommended)"
                        )

            if action_items:
                st.markdown("\n".join(action_items))
            else:
                st.success("No security or bug issues flagged by any model.")

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

    with tab_raw:
        for fr in reviewed:
            with st.expander(f"**{fr.filename}**"):
                if fr.error:
                    st.error(fr.error)
                else:
                    st.markdown(fr.raw)
