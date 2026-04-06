"""Core review engine — clones repos and reviews code with Gemma 4 via Ollama."""

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import ollama
from git import Repo

MODEL = "gemma4:26b"

# Patterns that indicate a section has no findings — LLMs vary their phrasing.
_CLEAN_PATTERNS = re.compile(
    r"none\s*found|no\s+(issues?|concerns?|problems?|bugs?|errors?)\s*(found|detected|identified|noted)?",
    re.IGNORECASE,
)

# Directories a local-path review must live under (empty = allow all).
# Reject clearly sensitive system paths.
_BLOCKED_PATHS = ("/etc", "/private/etc", "/var", "/private/var", "/System")
_BLOCKED_HOME_DIRS = (".ssh", ".gnupg", ".aws", ".config/gcloud")

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


def section_is_clean(text: str) -> bool:
    """Return True if a review section indicates no findings."""
    if not text:
        return True
    return bool(_CLEAN_PATTERNS.search(text))


def validate_source(source: str) -> str | None:
    """Return an error message if source is unsafe, else None."""
    if source.startswith(("http://", "https://")) or source.endswith(".git"):
        # Only allow well-formed GitHub/GitLab-style URLs
        if not re.match(r"https?://[\w.\-]+/[\w.\-/]+(?:\.git)?$", source):
            return "URL looks malformed. Please provide a standard Git repository URL."
        return None

    # Local path validation
    resolved = Path(source).expanduser().resolve()
    path_str = str(resolved)

    for blocked in _BLOCKED_PATHS:
        if path_str.startswith(blocked):
            return f"Refusing to review system path: {blocked}"

    home = Path.home()
    try:
        rel = resolved.relative_to(home)
        for blocked_dir in _BLOCKED_HOME_DIRS:
            if str(rel).startswith(blocked_dir):
                return f"Refusing to review sensitive directory: ~/{blocked_dir}"
    except ValueError:
        pass  # not under home — already checked _BLOCKED_PATHS

    if not resolved.is_dir():
        return f"Path does not exist or is not a directory: {source}"

    return None


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
    try:
        Repo.clone_from(url, str(tmp))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return tmp


def parse_review(raw: str) -> dict:
    """Parse the structured review response into sections.

    Handles varying markdown heading levels (##, ###, ####) and common
    phrasing variations from the LLM.
    """
    sections = {
        "summary": "", "security": "", "bugs": "",
        "style": "", "performance": "", "score": "",
    }
    # Map keywords found anywhere in a heading to a section key
    _keyword_to_section = [
        ("score", "score"),
        ("security", "security"),
        ("bug", "bugs"),
        ("logic error", "bugs"),
        ("style", "style"),
        ("best practice", "style"),
        ("performance", "performance"),
        ("summary", "summary"),
        ("overview", "summary"),
    ]
    current_key = None
    for line in raw.split("\n"):
        stripped = line.strip()
        # Detect any markdown heading (##, ###, ####, etc.)
        if re.match(r"^#{1,6}\s+", stripped):
            heading_text = stripped.lstrip("#").strip().lower()
            matched = False
            for keyword, section in _keyword_to_section:
                if keyword in heading_text:
                    current_key = section
                    matched = True
                    break
            if not matched:
                current_key = None
            continue
        if current_key:
            sections[current_key] += line + "\n"
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


def review_repo(source: str, max_files: int = 20, on_progress=None) -> RepoReview:
    """Review a repo from a URL or local path. Returns a RepoReview.

    Args:
        on_progress: Optional callback(current_index, total, filename) called
                     before each file review starts.

    Raises:
        ValueError: If the source fails input validation.
    """
    error = validate_source(source)
    if error:
        raise ValueError(error)

    cleanup = False
    is_remote = source.startswith(("http://", "https://")) or source.endswith(".git")

    if is_remote:
        if on_progress:
            on_progress(-1, 0, "Cloning repository...")
        root = clone_repo(source)
        cleanup = True
    else:
        root = Path(source).expanduser().resolve()

    try:
        files, skipped = collect_files(root)
        to_review = files[:max_files]
        total = len(to_review)
        review = RepoReview(repo_source=source, files_skipped=skipped)

        for i, f in enumerate(to_review):
            rel = str(f.relative_to(root))
            if on_progress:
                on_progress(i, total, rel)
            fr = review_file(f, root)
            review.files_reviewed.append(fr)

        return review
    finally:
        if cleanup:
            shutil.rmtree(root, ignore_errors=True)
