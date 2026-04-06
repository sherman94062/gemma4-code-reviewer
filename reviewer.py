"""Core review engine — clones repos and reviews code with Gemma 4 via Ollama."""

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import ollama
from git import Repo

MODEL = "gemma4:26b"

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".sql", ".sh", ".bash", ".yaml", ".yml",
    ".toml", ".json", ".html", ".css",
}

MAX_FILE_SIZE = 50_000  # characters — skip very large files

REVIEW_PROMPT = """\
You are an expert code reviewer. Analyze the following file and produce a structured review.

**File:** `{filename}`
**Language:** {language}

```
{code}
```

Provide your review in EXACTLY this format (use these exact headings):

## Summary
One paragraph summarizing what this file does and its overall quality.

## Security Issues
Bullet list of security concerns (injection, hardcoded secrets, auth issues, etc.). Say "None found." if clean.

## Bugs & Logic Errors
Bullet list of potential bugs, race conditions, off-by-one errors, etc. Say "None found." if clean.

## Style & Best Practices
Bullet list of style improvements, naming, idiomatic patterns, etc. Say "None found." if clean.

## Performance
Bullet list of performance concerns (N+1 queries, unnecessary allocations, etc.). Say "None found." if clean.

## Score
Give an overall score from 1-10 and a one-line justification.
"""


@dataclass
class FileReview:
    filename: str
    language: str
    summary: str = ""
    security: str = ""
    bugs: str = ""
    style: str = ""
    performance: str = ""
    score: str = ""
    raw: str = ""
    error: str = ""


@dataclass
class RepoReview:
    repo_source: str
    files_reviewed: list[FileReview] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)


def language_from_ext(ext: str) -> str:
    mapping = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".tsx": "TypeScript/React", ".jsx": "JavaScript/React",
        ".go": "Go", ".rs": "Rust", ".java": "Java", ".rb": "Ruby",
        ".php": "PHP", ".c": "C", ".cpp": "C++", ".h": "C/C++ Header",
        ".hpp": "C++ Header", ".cs": "C#", ".swift": "Swift",
        ".kt": "Kotlin", ".scala": "Scala", ".sql": "SQL",
        ".sh": "Shell", ".bash": "Bash", ".yaml": "YAML", ".yml": "YAML",
        ".toml": "TOML", ".json": "JSON", ".html": "HTML", ".css": "CSS",
    }
    return mapping.get(ext, "Unknown")


def collect_files(root: Path) -> tuple[list[Path], list[str]]:
    """Walk the repo and return (reviewable_files, skipped_files)."""
    reviewable, skipped = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # skip hidden dirs, node_modules, vendor, etc.
        parts = path.relative_to(root).parts
        if any(p.startswith(".") or p in ("node_modules", "vendor", "__pycache__", "dist", "build", ".git") for p in parts):
            continue
        if path.suffix not in SUPPORTED_EXTENSIONS:
            skipped.append(str(path.relative_to(root)))
            continue
        if path.stat().st_size > MAX_FILE_SIZE:
            skipped.append(f"{path.relative_to(root)} (too large)")
            continue
        reviewable.append(path)
    return reviewable, skipped


def clone_repo(url: str) -> Path:
    """Clone a git repo to a temp directory and return the path."""
    tmp = Path(tempfile.mkdtemp(prefix="gemma4_review_"))
    Repo.clone_from(url, str(tmp))
    return tmp


def parse_review(raw: str) -> dict:
    """Parse the structured review response into sections."""
    sections = {
        "summary": "", "security": "", "bugs": "",
        "style": "", "performance": "", "score": "",
    }
    heading_map = {
        "summary": "summary",
        "security issues": "security",
        "security": "security",
        "bugs & logic errors": "bugs",
        "bugs": "bugs",
        "style & best practices": "style",
        "style": "style",
        "performance": "performance",
        "score": "score",
    }
    current_key = None
    for line in raw.split("\n"):
        stripped = line.strip().lstrip("#").strip()
        lower = stripped.lower()
        if lower in heading_map:
            current_key = heading_map[lower]
            continue
        if current_key:
            sections[current_key] += line + "\n"
    # trim trailing whitespace
    return {k: v.strip() for k, v in sections.items()}


def review_file(filepath: Path, root: Path) -> FileReview:
    """Send a single file to Gemma 4 for review."""
    rel = str(filepath.relative_to(root))
    ext = filepath.suffix
    lang = language_from_ext(ext)
    code = filepath.read_text(errors="replace")

    fr = FileReview(filename=rel, language=lang)
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": REVIEW_PROMPT.format(
                    filename=rel, language=lang, code=code,
                ),
            }],
        )
        raw = response["message"]["content"]
        fr.raw = raw
        parsed = parse_review(raw)
        fr.summary = parsed["summary"]
        fr.security = parsed["security"]
        fr.bugs = parsed["bugs"]
        fr.style = parsed["style"]
        fr.performance = parsed["performance"]
        fr.score = parsed["score"]
    except Exception as e:
        fr.error = str(e)
    return fr


def review_repo(source: str, max_files: int = 20) -> RepoReview:
    """Review a repo from a URL or local path. Returns a RepoReview."""
    cleanup = False
    if source.startswith("http://") or source.startswith("https://") or source.endswith(".git"):
        root = clone_repo(source)
        cleanup = True
    else:
        root = Path(source).expanduser().resolve()

    try:
        files, skipped = collect_files(root)
        review = RepoReview(repo_source=source, files_skipped=skipped)

        for f in files[:max_files]:
            fr = review_file(f, root)
            review.files_reviewed.append(fr)

        return review
    finally:
        if cleanup:
            shutil.rmtree(root, ignore_errors=True)
