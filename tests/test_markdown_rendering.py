import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_LATEX = (r"\operatorname", r"\newcommand", r"\DeclareMathOperator")
INLINE_CODE_WITH_DOUBLE_PIPE = re.compile(r"`[^`]*\|\|[^`]*`")


def _markdown_documents():
    for path in ROOT.rglob("*.md"):
        yield path, path.read_text(encoding="utf-8")

    for path in ROOT.rglob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        markdown = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "markdown":
                markdown.extend(cell.get("source", []))
                markdown.append("\n")
        yield path, "".join(markdown)


def test_github_markdown_math_has_no_known_rendering_hazards():
    errors = []

    for path, text in _markdown_documents():
        rel = path.relative_to(ROOT)

        for macro in FORBIDDEN_LATEX:
            if macro in text:
                errors.append(f"{rel}: forbidden GitHub math macro {macro}")

        in_display_math = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()

            if stripped == "$$":
                in_display_math = not in_display_math
                continue

            # A bare '=' directly below a math line is parsed by Markdown as a
            # Setext H1 underline before MathJax gets a chance to render it.
            if in_display_math and stripped == "=":
                errors.append(
                    f"{rel}:{lineno}: bare '=' inside $$ block becomes a Setext heading"
                )

            # GFM tables can split inline code such as `||R||` at the pipe
            # characters, yielding a visibly corrupted one-line table.
            if stripped.startswith("|") and INLINE_CODE_WITH_DOUBLE_PIPE.search(line):
                errors.append(
                    f"{rel}:{lineno}: inline code containing '||' inside Markdown table"
                )

        if in_display_math:
            errors.append(f"{rel}: unclosed $$ display-math block")

    assert not errors, "Markdown rendering hazards:\n" + "\n".join(errors)
