"""Renderer — deterministic Jinja2 → Markdown / HTML, optional WeasyPrint PDF.

Zero LLM involvement. All values must be pre-computed in the Pydantic schema.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jinja2

from src.models.obj3_gemini.schemas.base_schema import BaseReport

logger = logging.getLogger(__name__)

# Default template directory relative to model-pipeline root
_DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "templates"

# Template file lookup — each report type maps to its native format only.
# daily/high_risk → Markdown; incident/final → HTML.
_TEMPLATE_MAP: dict[str, dict[str, str]] = {
    "daily":     {"md": "daily.md.j2"},
    "high_risk": {"md": "high_risk.md.j2"},
    "incident":  {"html": "incident.html.j2"},
    "final":     {"html": "final.html.j2"},
}


def get_template(report_type: str, fmt: str) -> str:
    """Return the template filename for a given report type and format.

    Parameters
    ----------
    report_type:
        One of ``"daily"``, ``"high_risk"``, ``"incident"``, ``"final"``.
    fmt:
        ``"md"`` or ``"html"``.

    Returns
    -------
    str
        Template filename.
    """
    try:
        return _TEMPLATE_MAP[report_type][fmt]
    except KeyError:
        raise ValueError(f"No template for report_type={report_type!r}, format={fmt!r}") from None


def _get_jinja_env(template_dir: Path, autoescape: bool = False) -> jinja2.Environment:
    """Create a Jinja2 environment with the given template directory."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=autoescape,
        undefined=jinja2.StrictUndefined,  # raises on missing vars — intentional
    )


def render_markdown(
    report: BaseReport,
    template_dir: Path | None = None,
) -> str:
    """Render a report as Markdown using Jinja2 (daily / high_risk).

    Parameters
    ----------
    report:
        Parsed Pydantic report model.
    template_dir:
        Directory containing Jinja2 templates. Defaults to ``templates/``.

    Returns
    -------
    str
        Rendered Markdown string.
    """
    template_dir = template_dir or _DEFAULT_TEMPLATE_DIR
    template_name = get_template(report.report_type, "md")
    env = _get_jinja_env(template_dir, autoescape=False)
    tmpl = env.get_template(template_name)
    return tmpl.render(**report.model_dump())


def render_html(
    report: BaseReport,
    template_dir: Path | None = None,
) -> str:
    """Render a report as HTML using Jinja2 (incident / final).

    Parameters
    ----------
    report:
        Parsed Pydantic report model.
    template_dir:
        Directory containing Jinja2 templates. Defaults to ``templates/``.

    Returns
    -------
    str
        Rendered HTML string.
    """
    template_dir = template_dir or _DEFAULT_TEMPLATE_DIR
    template_name = get_template(report.report_type, "html")
    env = _get_jinja_env(template_dir, autoescape=True)
    tmpl = env.get_template(template_name)
    return tmpl.render(**report.model_dump())


def markdown_to_html(md_str: str) -> str:
    """Convert a Markdown string to HTML via ``python-markdown``."""
    import markdown
    return markdown.markdown(md_str, extensions=["tables", "fenced_code"])


def render_pdf(html_str: str, css_string: str = "") -> bytes:
    """Convert an HTML string to PDF via WeasyPrint.

    Called only on admin request. WeasyPrint is an optional dependency.

    Parameters
    ----------
    html_str:
        Rendered HTML content.
    css_string:
        Optional CSS to apply.

    Returns
    -------
    bytes
        PDF file content.
    """
    try:
        from weasyprint import CSS, HTML
    except ImportError as exc:
        raise ImportError(
            "WeasyPrint is required for PDF generation. "
            "Install with: pip install weasyprint"
        ) from exc

    stylesheets = [CSS(string=css_string)] if css_string else []
    return HTML(string=html_str).write_pdf(stylesheets=stylesheets)
