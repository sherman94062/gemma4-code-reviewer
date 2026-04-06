"""Gemma 4 Code Review Agent — Streamlit UI."""

import streamlit as st
from reviewer import review_repo, RepoReview, FileReview

st.set_page_config(page_title="Gemma 4 Code Reviewer", page_icon="🔍", layout="wide")

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🔍 Gemma 4 Code Review Agent")
st.caption("Local AI-powered code reviews — your code never leaves your machine.")

# ── Sidebar: input ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Repository")
    source = st.text_input(
        "GitHub URL or local path",
        placeholder="https://github.com/user/repo or /path/to/project",
    )
    max_files = st.slider("Max files to review", 1, 50, 10)
    run = st.button("Run Review", type="primary", use_container_width=True)

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Clones the repo (or reads local path)\n"
        "2. Extracts reviewable source files\n"
        "3. Sends each file to **Gemma 4 27B** running locally via Ollama\n"
        "4. Parses structured feedback into categories"
    )

# ── Run review ──────────────────────────────────────────────────────────────
if run and source:
    progress_bar = st.progress(0)
    status_text = st.empty()

    def on_progress(current, total, filename):
        if current == -1:
            status_text.markdown(f"**Cloning repository...**")
            progress_bar.progress(0)
        else:
            pct = current / total
            progress_bar.progress(pct)
            status_text.markdown(
                f"**Reviewing file {current + 1} of {total}:** `{filename}`"
            )

    result: RepoReview = review_repo(source, max_files=max_files, on_progress=on_progress)

    progress_bar.progress(1.0)
    status_text.markdown("**Review complete!**")
    st.session_state["result"] = result

if "result" not in st.session_state:
    st.info("Enter a GitHub URL or local path in the sidebar and click **Run Review**.")
    st.stop()

result: RepoReview = st.session_state["result"]

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
    if fr.security and "none found" not in fr.security.lower():
        security_count += 1
    if fr.bugs and "none found" not in fr.bugs.lower():
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
    if fr.security and "none found" not in fr.security.lower():
        return "🔴"
    if fr.bugs and "none found" not in fr.bugs.lower():
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
            if text and "none found" not in text.lower():
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
