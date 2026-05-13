from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def _default_data_root() -> Path:
    configured_root = (os.getenv("ML_APP_DATA_ROOT") or "").strip()
    if configured_root:
        return Path(configured_root).expanduser()

    if getattr(sys, "frozen", False):
        local_app_data = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        base_root = Path(local_app_data).expanduser() if local_app_data else Path.home()
        return base_root / "MLStudio" / "backend"

    return BASE_DIR / "data"


DEFAULT_DATA_ROOT = _default_data_root()
DEFAULT_DB_PATH = DEFAULT_DATA_ROOT / "ml_studio.db"
DEFAULT_UPLOAD_DIR = DEFAULT_DATA_ROOT / "uploads"
DEFAULT_PROJECT_SNAPSHOT_DIR = DEFAULT_DATA_ROOT / "project_state"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_project_name(project_name: str | None) -> str | None:
    normalized = (project_name or "").strip()
    return normalized or None


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=_json_default)


def _json_loads(payload: str | None) -> Any:
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _safe_filename(filename: str) -> str:
    raw_name = Path(filename or "dataset").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._") or "dataset"


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._") or "project"


def _extract_shape(dataset: dict[str, Any] | None) -> tuple[int, int]:
    if not dataset:
        return 0, 0

    shape = dataset.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return 0, 0

    try:
        row_count = max(int(shape[0]), 0)
    except (TypeError, ValueError):
        row_count = 0

    try:
        column_count = max(int(shape[1]), 0)
    except (TypeError, ValueError):
        column_count = 0

    return row_count, column_count


def get_db_path() -> Path:
    configured = (os.getenv("ML_APP_DB_PATH") or "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_DB_PATH


def get_upload_dir() -> Path:
    configured = (os.getenv("ML_APP_UPLOAD_DIR") or "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_UPLOAD_DIR


def get_project_snapshot_dir() -> Path:
    configured = (os.getenv("ML_APP_PROJECT_STATE_DIR") or "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_PROJECT_SNAPSHOT_DIR


def _connect() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database() -> dict[str, Any]:
    get_upload_dir().mkdir(parents=True, exist_ok=True)
    get_project_snapshot_dir().mkdir(parents=True, exist_ok=True)

    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                notes TEXT DEFAULT '',
                latest_dataset_filename TEXT,
                latest_dataset_path TEXT,
                latest_row_count INTEGER DEFAULT 0,
                latest_column_count INTEGER DEFAULT 0,
                latest_snapshot_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                project_name TEXT,
                original_filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                file_size INTEGER NOT NULL DEFAULT 0,
                row_count INTEGER NOT NULL DEFAULT 0,
                column_count INTEGER NOT NULL DEFAULT 0,
                dataset_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                project_name TEXT,
                task_type TEXT,
                model_type TEXT,
                model_label TEXT,
                target_column TEXT,
                metrics_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                project_name TEXT,
                analysis_type TEXT,
                title TEXT,
                summary TEXT,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                project_name TEXT,
                title TEXT,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS project_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                project_name TEXT NOT NULL,
                project_notes TEXT DEFAULT '',
                snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """
        )
        connection.commit()

    return get_database_info()


def _row_to_project_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None

    return {
        "id": row["id"],
        "name": row["name"],
        "notes": row["notes"] or "",
        "latest_dataset_filename": row["latest_dataset_filename"],
        "latest_dataset_path": row["latest_dataset_path"],
        "latest_row_count": row["latest_row_count"] or 0,
        "latest_column_count": row["latest_column_count"] or 0,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _get_project(connection: sqlite3.Connection, project_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
            id,
            name,
            notes,
            latest_dataset_filename,
            latest_dataset_path,
            latest_row_count,
            latest_column_count,
            created_at,
            updated_at
        FROM projects
        WHERE id = ?
        """,
        (project_id,),
    ).fetchone()
    return _row_to_project_dict(row)


def _ensure_project(
    connection: sqlite3.Connection,
    project_name: str,
    project_notes: str | None = None,
    dataset: dict[str, Any] | None = None,
    latest_snapshot: dict[str, Any] | None = None,
    latest_dataset_path: str | None = None,
) -> dict[str, Any]:
    normalized_name = project_name.strip()
    current_time = _now()
    row_count, column_count = _extract_shape(dataset)
    dataset_filename = dataset.get("filename") if isinstance(dataset, dict) else None
    snapshot_json = _json_dumps(latest_snapshot) if latest_snapshot else None

    existing = connection.execute(
        """
        SELECT
            id,
            notes,
            latest_dataset_filename,
            latest_dataset_path,
            latest_row_count,
            latest_column_count
        FROM projects
        WHERE name = ?
        """,
        (normalized_name,),
    ).fetchone()

    if existing is None:
        cursor = connection.execute(
            """
            INSERT INTO projects (
                name,
                notes,
                latest_dataset_filename,
                latest_dataset_path,
                latest_row_count,
                latest_column_count,
                latest_snapshot_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_name,
                project_notes or "",
                dataset_filename,
                latest_dataset_path,
                row_count,
                column_count,
                snapshot_json,
                current_time,
                current_time,
            ),
        )
        return _get_project(connection, int(cursor.lastrowid)) or {
            "id": int(cursor.lastrowid),
            "name": normalized_name,
            "notes": project_notes or "",
            "latest_dataset_filename": dataset_filename,
            "latest_dataset_path": latest_dataset_path,
            "latest_row_count": row_count,
            "latest_column_count": column_count,
            "created_at": current_time,
            "updated_at": current_time,
        }

    next_notes = project_notes if project_notes is not None else existing["notes"]
    next_dataset_filename = dataset_filename or existing["latest_dataset_filename"]
    next_dataset_path = latest_dataset_path or existing["latest_dataset_path"]
    next_row_count = row_count or int(existing["latest_row_count"] or 0)
    next_column_count = column_count or int(existing["latest_column_count"] or 0)

    connection.execute(
        """
        UPDATE projects
        SET
            notes = ?,
            latest_dataset_filename = ?,
            latest_dataset_path = ?,
            latest_row_count = ?,
            latest_column_count = ?,
            latest_snapshot_json = COALESCE(?, latest_snapshot_json),
            updated_at = ?
        WHERE id = ?
        """,
        (
            next_notes or "",
            next_dataset_filename,
            next_dataset_path,
            next_row_count,
            next_column_count,
            snapshot_json,
            current_time,
            existing["id"],
        ),
    )
    return _get_project(connection, int(existing["id"])) or {
        "id": int(existing["id"]),
        "name": normalized_name,
        "notes": next_notes or "",
        "latest_dataset_filename": next_dataset_filename,
        "latest_dataset_path": next_dataset_path,
        "latest_row_count": next_row_count,
        "latest_column_count": next_column_count,
        "created_at": current_time,
        "updated_at": current_time,
    }


def _save_uploaded_file(filename: str, contents: bytes) -> Path:
    upload_dir = get_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved_path = upload_dir / f"{timestamp}_{safe_name}"
    saved_path.write_bytes(contents)
    return saved_path


def save_dataframe_snapshot(
    project_name: str,
    dataframe: pd.DataFrame,
    source_filename: str | None = None,
) -> str:
    snapshot_dir = get_project_snapshot_dir() / _safe_slug(project_name)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    source_stem = Path(source_filename or "dataset.csv").stem
    snapshot_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{_safe_slug(source_stem)}.csv"
    snapshot_path = snapshot_dir / snapshot_name
    dataframe.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
    return str(snapshot_path)


def get_database_info() -> dict[str, Any]:
    db_path = get_db_path()
    info = {
        "enabled": True,
        "engine": "sqlite",
        "path": str(db_path),
        "exists": db_path.exists(),
        "size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "project_count": 0,
        "dataset_upload_count": 0,
        "training_run_count": 0,
        "analysis_run_count": 0,
        "report_count": 0,
        "snapshot_count": 0,
    }

    if not db_path.exists():
        return info

    try:
        with _connect() as connection:
            info["project_count"] = int(connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0])
            info["dataset_upload_count"] = int(connection.execute("SELECT COUNT(*) FROM dataset_uploads").fetchone()[0])
            info["training_run_count"] = int(connection.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0])
            info["analysis_run_count"] = int(connection.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0])
            info["report_count"] = int(connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0])
            info["snapshot_count"] = int(connection.execute("SELECT COUNT(*) FROM project_snapshots").fetchone()[0])
    except sqlite3.Error:
        return info

    return info


def record_dataset_upload(
    project_name: str | None,
    filename: str,
    contents: bytes,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    saved_path = _save_uploaded_file(filename, contents)
    row_count, column_count = _extract_shape(dataset)
    normalized_name = _normalize_project_name(project_name)

    with _connect() as connection:
        project = None
        project_id = None
        if normalized_name:
            project = _ensure_project(
                connection,
                normalized_name,
                dataset=dataset,
                latest_dataset_path=str(saved_path),
            )
            project_id = project["id"]

        cursor = connection.execute(
            """
            INSERT INTO dataset_uploads (
                project_id,
                project_name,
                original_filename,
                stored_path,
                file_size,
                row_count,
                column_count,
                dataset_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                normalized_name,
                filename,
                str(saved_path),
                len(contents),
                row_count,
                column_count,
                _json_dumps(dataset),
                _now(),
            ),
        )
        connection.commit()

        return {
            "id": int(cursor.lastrowid),
            "project": project,
            "stored_path": str(saved_path),
            "file_size": len(contents),
            "row_count": row_count,
            "column_count": column_count,
        }


def save_training_run(
    project_name: str | None,
    dataset: dict[str, Any] | None,
    training_result: dict[str, Any],
) -> dict[str, Any]:
    normalized_name = _normalize_project_name(project_name)

    with _connect() as connection:
        project = None
        project_id = None
        if normalized_name:
            project = _ensure_project(connection, normalized_name, dataset=dataset)
            project_id = project["id"]

        cursor = connection.execute(
            """
            INSERT INTO training_runs (
                project_id,
                project_name,
                task_type,
                model_type,
                model_label,
                target_column,
                metrics_json,
                result_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                normalized_name,
                training_result.get("task_type"),
                training_result.get("model_type"),
                training_result.get("model_label"),
                training_result.get("target_column"),
                _json_dumps(training_result.get("metrics", {})),
                _json_dumps(training_result),
                _now(),
            ),
        )
        connection.commit()

        return {
            "id": int(cursor.lastrowid),
            "project": project,
        }


def save_analysis_run(
    project_name: str | None,
    dataset: dict[str, Any] | None,
    analysis_result: dict[str, Any],
) -> dict[str, Any]:
    normalized_name = _normalize_project_name(project_name)

    with _connect() as connection:
        project = None
        project_id = None
        if normalized_name:
            project = _ensure_project(connection, normalized_name, dataset=dataset)
            project_id = project["id"]

        cursor = connection.execute(
            """
            INSERT INTO analysis_runs (
                project_id,
                project_name,
                analysis_type,
                title,
                summary,
                result_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                normalized_name,
                analysis_result.get("analysis_type"),
                analysis_result.get("title"),
                analysis_result.get("summary"),
                _json_dumps(analysis_result),
                _now(),
            ),
        )
        connection.commit()

        return {
            "id": int(cursor.lastrowid),
            "project": project,
        }


def save_report_record(
    project_name: str | None,
    dataset: dict[str, Any] | None,
    report_result: dict[str, Any],
) -> dict[str, Any]:
    normalized_name = _normalize_project_name(project_name)

    with _connect() as connection:
        project = None
        project_id = None
        if normalized_name:
            project = _ensure_project(connection, normalized_name, dataset=dataset)
            project_id = project["id"]

        cursor = connection.execute(
            """
            INSERT INTO reports (
                project_id,
                project_name,
                title,
                report_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                project_id,
                normalized_name,
                report_result.get("title"),
                _json_dumps(report_result),
                _now(),
            ),
        )
        connection.commit()

        return {
            "id": int(cursor.lastrowid),
            "project": project,
        }


def save_project_snapshot(
    project_name: str,
    project_notes: str | None,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    normalized_name = project_name.strip()
    if not normalized_name:
        raise ValueError("项目名称不能为空。")

    dataset = snapshot.get("dataset") if isinstance(snapshot, dict) else None

    with _connect() as connection:
        project = _ensure_project(
            connection,
            normalized_name,
            project_notes=project_notes,
            dataset=dataset if isinstance(dataset, dict) else None,
            latest_snapshot=snapshot,
            latest_dataset_path=snapshot.get("dataset_storage_path"),
        )
        cursor = connection.execute(
            """
            INSERT INTO project_snapshots (
                project_id,
                project_name,
                project_notes,
                snapshot_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                project["id"],
                normalized_name,
                project_notes or "",
                _json_dumps(snapshot),
                _now(),
            ),
        )
        connection.commit()

        return {
            "snapshot_id": int(cursor.lastrowid),
            "project": project,
        }


def get_project_detail(project_id: int) -> dict[str, Any] | None:
    with _connect() as connection:
        project_row = connection.execute(
            """
            SELECT
                p.id,
                p.name,
                p.notes,
                p.latest_dataset_filename,
                p.latest_dataset_path,
                p.latest_row_count,
                p.latest_column_count,
                p.created_at,
                p.updated_at,
                (SELECT COUNT(*) FROM dataset_uploads d WHERE d.project_id = p.id) AS dataset_upload_count,
                (SELECT COUNT(*) FROM training_runs t WHERE t.project_id = p.id) AS training_run_count,
                (SELECT COUNT(*) FROM analysis_runs a WHERE a.project_id = p.id) AS analysis_run_count,
                (SELECT COUNT(*) FROM reports r WHERE r.project_id = p.id) AS report_count,
                (SELECT COUNT(*) FROM project_snapshots s WHERE s.project_id = p.id) AS snapshot_count
            FROM projects p
            WHERE p.id = ?
            """,
            (project_id,),
        ).fetchone()

        if project_row is None:
            return None

        snapshot_row = connection.execute(
            """
            SELECT id, project_notes, snapshot_json, created_at
            FROM project_snapshots
            WHERE project_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()

        latest_upload_row = connection.execute(
            """
            SELECT id, original_filename, stored_path, dataset_json, created_at
            FROM dataset_uploads
            WHERE project_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()

    project = {
        "id": project_row["id"],
        "name": project_row["name"],
        "notes": project_row["notes"] or "",
        "latest_dataset_filename": project_row["latest_dataset_filename"],
        "latest_dataset_path": project_row["latest_dataset_path"],
        "latest_row_count": project_row["latest_row_count"] or 0,
        "latest_column_count": project_row["latest_column_count"] or 0,
        "created_at": project_row["created_at"],
        "updated_at": project_row["updated_at"],
        "dataset_upload_count": project_row["dataset_upload_count"] or 0,
        "training_run_count": project_row["training_run_count"] or 0,
        "analysis_run_count": project_row["analysis_run_count"] or 0,
        "report_count": project_row["report_count"] or 0,
        "snapshot_count": project_row["snapshot_count"] or 0,
    }

    latest_snapshot = None
    if snapshot_row is not None:
        latest_snapshot = {
            "id": snapshot_row["id"],
            "project_notes": snapshot_row["project_notes"] or "",
            "created_at": snapshot_row["created_at"],
            "data": _json_loads(snapshot_row["snapshot_json"]) or {},
        }

    latest_upload = None
    if latest_upload_row is not None:
        latest_upload = {
            "id": latest_upload_row["id"],
            "original_filename": latest_upload_row["original_filename"],
            "stored_path": latest_upload_row["stored_path"],
            "created_at": latest_upload_row["created_at"],
            "dataset": _json_loads(latest_upload_row["dataset_json"]) or {},
        }

    return {
        "project": project,
        "latest_snapshot": latest_snapshot,
        "latest_upload": latest_upload,
    }


def list_recent_projects(limit: int = 12) -> list[dict[str, Any]]:
    safe_limit = min(max(int(limit), 1), 50)

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT
                p.id,
                p.name,
                p.notes,
                p.latest_dataset_filename,
                p.latest_dataset_path,
                p.latest_row_count,
                p.latest_column_count,
                p.created_at,
                p.updated_at,
                (SELECT COUNT(*) FROM dataset_uploads d WHERE d.project_id = p.id) AS dataset_upload_count,
                (SELECT COUNT(*) FROM training_runs t WHERE t.project_id = p.id) AS training_run_count,
                (SELECT COUNT(*) FROM analysis_runs a WHERE a.project_id = p.id) AS analysis_run_count,
                (SELECT COUNT(*) FROM reports r WHERE r.project_id = p.id) AS report_count,
                (SELECT COUNT(*) FROM project_snapshots s WHERE s.project_id = p.id) AS snapshot_count
            FROM projects p
            ORDER BY p.updated_at DESC, p.id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "notes": row["notes"] or "",
            "latest_dataset_filename": row["latest_dataset_filename"],
            "latest_dataset_path": row["latest_dataset_path"],
            "latest_row_count": row["latest_row_count"] or 0,
            "latest_column_count": row["latest_column_count"] or 0,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "dataset_upload_count": row["dataset_upload_count"] or 0,
            "training_run_count": row["training_run_count"] or 0,
            "analysis_run_count": row["analysis_run_count"] or 0,
            "report_count": row["report_count"] or 0,
            "snapshot_count": row["snapshot_count"] or 0,
        }
        for row in rows
    ]
