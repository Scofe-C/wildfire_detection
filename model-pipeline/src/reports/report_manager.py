"""Report manager — persist reports locally and sync to GCS.

Handles file naming conventions, directory structure, and optional
GCS upload via subprocess ``gsutil``.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReportIndex:
    """Lightweight metadata for a saved report (no full content loaded)."""

    report_type: str
    filename: str
    json_path: Path
    rendered_path: Path | None
    created_at: datetime


# ---------------------------------------------------------------------------
# Naming convention
# ---------------------------------------------------------------------------

_TYPE_PREFIX: dict[str, str] = {
    "daily": "DailyReport",
    "high_risk": "HighRiskReport",
    "incident": "IncidentReport",
    "final": "FinalReport",
}


def make_filename(report_type: str, dt: datetime) -> str:
    """Generate report filename stem: ``ReportType_YYYYMMDD_HHMM``.

    Parameters
    ----------
    report_type:
        One of ``"daily"``, ``"high_risk"``, ``"incident"``, ``"final"``.
    dt:
        Timestamp for the filename.

    Returns
    -------
    str
        Filename stem (no extension).
    """
    prefix = _TYPE_PREFIX.get(report_type, report_type.title() + "Report")
    return f"{prefix}_{dt.strftime('%Y%m%d_%H%M')}"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_report(
    report_json: str,
    rendered_content: str,
    report_type: str,
    incident_id: str,
    dt: datetime,
    fmt: str,
    output_dir: Path,
) -> tuple[Path, Path | None]:
    """Write ``.json`` and optionally ``.md`` / ``.html`` files side by side.

    Creates subdirectory (e.g. ``daily/``) if it doesn't exist.
    When ``rendered_content`` is empty, only the JSON file is written.

    Parameters
    ----------
    report_json:
        Serialised JSON string.
    rendered_content:
        Rendered Markdown or HTML string. Empty string skips rendered file.
    report_type:
        One of ``"daily"``, ``"high_risk"``, ``"incident"``, ``"final"``.
    incident_id:
        UUID for this report (included in JSON, not in filename).
    dt:
        Timestamp for the filename.
    fmt:
        ``"md"`` or ``"html"`` — determines rendered file extension.
    output_dir:
        Root output directory (e.g. ``reports/disaster_reports``).

    Returns
    -------
    tuple[Path, Path | None]
        ``(json_path, rendered_path)`` — ``rendered_path`` is ``None``
        when no rendered content was provided (JSON-only mode).
    """
    subdir = output_dir / report_type
    subdir.mkdir(parents=True, exist_ok=True)

    stem = make_filename(report_type, dt)
    json_path = subdir / f"{stem}.json"

    json_path.write_text(report_json, encoding="utf-8")

    rendered_path: Path | None = None
    if rendered_content:
        rendered_path = subdir / f"{stem}.{fmt}"
        rendered_path.write_text(rendered_content, encoding="utf-8")
        logger.info("Saved report: %s + %s", json_path.name, rendered_path.name)
    else:
        logger.info("Saved report (JSON only): %s", json_path.name)

    return json_path, rendered_path


# ---------------------------------------------------------------------------
# GCS sync (non-blocking, best-effort)
# ---------------------------------------------------------------------------

def sync_to_gcs(
    local_paths: list[Path],
    gcs_bucket: str,
    gcs_prefix: str = "",
) -> list[str]:
    """Upload local files to GCS via ``gsutil cp``.

    Non-blocking: GCS failure is logged but does NOT raise.

    Returns
    -------
    list[str]
        List of ``gs://`` URIs for successfully uploaded files.
    """
    if not gcs_bucket:
        logger.debug("GCS bucket not configured — skipping sync.")
        return []

    uploaded: list[str] = []
    for path in local_paths:
        gcs_uri = f"gs://{gcs_bucket}/{gcs_prefix}{path.parent.name}/{path.name}"
        try:
            subprocess.run(
                ["gsutil", "cp", str(path), gcs_uri],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            uploaded.append(gcs_uri)
            logger.info("Uploaded to GCS: %s", gcs_uri)
        except FileNotFoundError:
            logger.warning("gsutil not found — skipping GCS sync for %s", path.name)
            break
        except subprocess.SubprocessError as exc:
            logger.warning("GCS sync failed for %s: %s", path.name, exc)

    return uploaded


# ---------------------------------------------------------------------------
# Listing / indexing
# ---------------------------------------------------------------------------

def list_reports(
    report_type: str,
    output_dir: Path,
    date_range: tuple[date, date] | None = None,
) -> list[ReportIndex]:
    """List saved reports without loading full content.

    Parameters
    ----------
    report_type:
        Filter by report type subdirectory.
    output_dir:
        Root output directory.
    date_range:
        Optional ``(start, end)`` date filter (inclusive).

    Returns
    -------
    list[ReportIndex]
    """
    subdir = output_dir / report_type
    if not subdir.is_dir():
        return []

    results: list[ReportIndex] = []
    for json_file in sorted(subdir.glob("*.json")):
        stem = json_file.stem
        # Try to parse date from filename: ReportType_YYYYMMDD_HHMM
        parts = stem.rsplit("_", 2)
        if len(parts) >= 3:
            try:
                dt = datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M")
            except ValueError:
                dt = datetime.fromtimestamp(json_file.stat().st_mtime)
        else:
            dt = datetime.fromtimestamp(json_file.stat().st_mtime)

        # Apply date filter
        if date_range and (dt.date() < date_range[0] or dt.date() > date_range[1]):
            continue

        # Find companion rendered file
        rendered = None
        for ext in (".md", ".html"):
            candidate = json_file.with_suffix(ext)
            if candidate.exists():
                rendered = candidate
                break

        results.append(ReportIndex(
            report_type=report_type,
            filename=stem,
            json_path=json_file,
            rendered_path=rendered,
            created_at=dt,
        ))

    return results


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_old_reports(
    output_dir: Path,
    max_age_days: int = 90,
    keep_finals: bool = True,
) -> list[Path]:
    """Delete report files (JSON + rendered) older than ``max_age_days``.

    Optionally preserves ``final/`` subdirectory reports for compliance.

    Parameters
    ----------
    output_dir:
        Root output directory (e.g. ``reports/disaster_reports``).
    max_age_days:
        Maximum age in days. Files older than this are deleted.
    keep_finals:
        If True, never delete files in the ``final/`` subdirectory.

    Returns
    -------
    list[Path]
        Paths of deleted files.
    """
    if not output_dir.is_dir():
        return []

    cutoff = time.time() - (max_age_days * 86400)
    deleted: list[Path] = []

    for json_file in output_dir.rglob("*.json"):
        if "review_manifest" in json_file.name or "incident_state" in json_file.name:
            continue
        if keep_finals and "final" in json_file.parent.name:
            continue

        if json_file.stat().st_mtime < cutoff:
            # Delete JSON and any companion rendered file
            for ext in (".md", ".html"):
                companion = json_file.with_suffix(ext)
                if companion.exists():
                    companion.unlink()
                    deleted.append(companion)
            json_file.unlink()
            deleted.append(json_file)

    if deleted:
        logger.info("Cleanup: deleted %d old report files (max_age=%dd)", len(deleted), max_age_days)
    return deleted


def delete_report(report_id: str, output_dir: Path) -> list[Path]:
    """Delete a single report (JSON + rendered) by its ID (filename stem).

    Returns list of deleted paths, or empty if not found.
    """
    deleted: list[Path] = []
    matches = list(output_dir.rglob(f"{report_id}.json"))
    if not matches:
        return deleted

    json_path = matches[0]
    for ext in (".json", ".md", ".html"):
        target = json_path.with_suffix(ext)
        if target.exists():
            target.unlink()
            deleted.append(target)

    if deleted:
        logger.info("Deleted report %s (%d files)", report_id, len(deleted))
    return deleted
