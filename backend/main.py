from typing import Any, Literal
import io
import os

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_fscore_support,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from db import (
    get_database_info,
    get_project_detail,
    initialize_database,
    list_recent_projects,
    record_dataset_upload,
    save_analysis_run,
    save_dataframe_snapshot,
    save_project_snapshot,
    save_report_record,
    save_training_run,
)
from studio_services import (
    apply_data_process,
    generate_report,
    run_statistical_analysis,
    serialize_dataset,
)

BACKEND_VERSION = "2026.04.16-r4"
DEFAULT_PORT = int(os.getenv("ML_APP_PORT", "8010"))

app = FastAPI(title="ML Pro Backend", version=BACKEND_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_df: pd.DataFrame | None = None
original_df: pd.DataFrame | None = None
current_filename: str | None = None
data_history: list[dict[str, Any]] = []
current_project_name: str | None = None
current_project_notes: str | None = None


class TrainParams(BaseModel):
    project_name: str | None = None
    target_column: str
    feature_columns: list[str]
    model_type: str
    task_type: Literal["classification", "regression"]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class DataProcessParams(BaseModel):
    action: str
    columns: list[str] = Field(default_factory=list)
    strategy: str | None = None
    fill_value: str | float | int | None = None
    method: str | None = None
    threshold: float | None = None
    target_dtype: str | None = None
    filter_expression: str | None = None
    new_column: str | None = None
    expression: str | None = None


class AnalysisParams(BaseModel):
    project_name: str | None = None
    analysis_type: str
    columns: list[str] = Field(default_factory=list)
    target_column: str | None = None
    group_column: str | None = None
    row_column: str | None = None
    column_column: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    n_clusters: int | None = None
    n_components: int | None = None
    n_factors: int | None = None


class ReportParams(BaseModel):
    project_name: str | None = None
    dataset: dict[str, Any] = Field(default_factory=dict)
    processing_history: list[dict[str, Any]] = Field(default_factory=list)
    training_result: dict[str, Any] = Field(default_factory=dict)
    analysis_results: list[dict[str, Any]] = Field(default_factory=list)


class SnapshotParams(BaseModel):
    project_name: str
    project_notes: str | None = None
    dataset: dict[str, Any] = Field(default_factory=dict)
    processing_history: list[dict[str, Any]] = Field(default_factory=list)
    training_result: dict[str, Any] = Field(default_factory=dict)
    analysis_results: list[dict[str, Any]] = Field(default_factory=list)
    report_result: dict[str, Any] = Field(default_factory=dict)
    workspace_state: dict[str, Any] = Field(default_factory=dict)


def _get_model_label(model_type: str, task_type: str) -> str:
    if model_type == "RandomForest":
        return "随机森林"
    if model_type == "SVM":
        return "支持向量机"
    if model_type == "LogisticRegression":
        return "逻辑回归" if task_type == "classification" else "线性回归"
    return model_type


def _get_task_label(task_type: str) -> str:
    return "分类" if task_type == "classification" else "回归"


def _get_success_message(model_type: str, task_type: str) -> str:
    model_label = _get_model_label(model_type, task_type)
    task_label = _get_task_label(task_type)

    if model_label.endswith(task_label):
        return f"已完成{model_label}训练"

    return f"已完成{model_label}{task_label}训练"


def _parse_optional_int(value: Any, field_name: str) -> int | None:
    if value in (None, ""):
        return None

    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} 必须大于 0。")
    return parsed


def _parse_optional_float(value: Any, field_name: str) -> float | None:
    if value in (None, ""):
        return None

    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} 必须大于 0。")
    return parsed


def _looks_like_continuous_target(target: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(target):
        return False

    numeric_target = pd.to_numeric(target, errors="coerce").dropna()
    if numeric_target.empty:
        return False

    unique_count = int(numeric_target.nunique(dropna=True))
    sample_count = int(len(numeric_target))
    unique_ratio = unique_count / max(sample_count, 1)

    if unique_count <= 12:
        return False

    return unique_ratio >= 0.2 or unique_count >= 50


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise HTTPException(status_code=400, detail="没有可用的特征列，请重新选择字段。")

    return ColumnTransformer(transformers=transformers)


def _compact_points(points: list[dict[str, Any]], max_points: int = 60) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points

    step = len(points) / max_points
    return [points[int(index * step)] for index in range(max_points)]


def _format_feature_name(name: str) -> str:
    cleaned = name.replace("numeric__", "").replace("categorical__", "")
    return cleaned.replace("__", " / ")


def _build_metric_bars(task_type: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    if task_type == "classification":
        return [
            {"name": "准确率", "value": round(metrics["accuracy"] * 100, 2)},
            {"name": "精确率", "value": round(metrics["precision"] * 100, 2)},
            {"name": "召回率", "value": round(metrics["recall"] * 100, 2)},
            {"name": "F1", "value": round(metrics["f1_score"] * 100, 2)},
        ]

    return [
        {"name": "RMSE", "value": metrics["rmse"]},
        {"name": "MAE", "value": metrics["mae"]},
        {"name": "R2", "value": metrics["r2_score"]},
    ]


def _build_feature_effects(pipeline: Pipeline) -> list[dict[str, Any]]:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        return []

    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefficients = model.coef_
        if getattr(coefficients, "ndim", 1) > 1:
            values = abs(coefficients).mean(axis=0)
        else:
            values = abs(coefficients)

    if values is None:
        return []

    feature_effects = [
        {
            "feature": _format_feature_name(str(name)),
            "value": round(float(value), 6),
        }
        for name, value in zip(feature_names, values)
    ]

    feature_effects.sort(key=lambda item: item["value"], reverse=True)
    return feature_effects[:12]


def _build_model(task_type: str, model_type: str, hyperparameters: dict[str, Any]):
    if model_type == "RandomForest":
        n_estimators = _parse_optional_int(hyperparameters.get("n_estimators", 100), "n_estimators") or 100
        max_depth = _parse_optional_int(hyperparameters.get("max_depth"), "max_depth")

        if task_type == "classification":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
            )

        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )

    if model_type == "SVM":
        c_value = _parse_optional_float(hyperparameters.get("C", 1.0), "C") or 1.0
        kernel = str(hyperparameters.get("kernel", "rbf"))
        gamma_value = hyperparameters.get("gamma", "scale")

        if gamma_value not in ("scale", "auto", None, ""):
            gamma_value = float(gamma_value)

        if task_type == "classification":
            return SVC(C=c_value, kernel=kernel, gamma=gamma_value)

        return SVR(C=c_value, kernel=kernel, gamma=gamma_value)

    if model_type == "LogisticRegression":
        if task_type == "classification":
            c_value = _parse_optional_float(hyperparameters.get("C", 1.0), "C") or 1.0
            max_iter = _parse_optional_int(hyperparameters.get("max_iter", 1000), "max_iter") or 1000
            return LogisticRegression(C=c_value, max_iter=max_iter, random_state=42)

        return LinearRegression()

    raise HTTPException(status_code=400, detail=f"暂不支持的模型类型：{model_type}")


@app.on_event("startup")
def on_startup():
    initialize_database()


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "ML Backend is running",
        "version": BACKEND_VERSION,
        "default_port": DEFAULT_PORT,
    }


@app.get("/api/meta")
def get_meta():
    return {
        "version": BACKEND_VERSION,
        "supported_task_types": ["classification", "regression"],
        "supported_modules": ["data_prepare", "modeling", "statistics", "reporting", "project"],
        "persistence_mode": "sqlite_local",
        "database": get_database_info(),
        "default_port": DEFAULT_PORT,
    }


def _require_current_df() -> pd.DataFrame:
    if current_df is None:
        raise HTTPException(status_code=400, detail="当前还没有载入数据集，请先上传数据。")
    return current_df


def _normalize_project_name(project_name: str | None) -> str | None:
    normalized = (project_name or "").strip()
    return normalized or None


def _current_dataset_payload() -> dict[str, Any]:
    if current_df is None:
        return {}
    return serialize_dataset(current_df, current_filename, data_history)


def _load_dataframe_from_path(dataset_path: str) -> pd.DataFrame:
    path = os.path.abspath(dataset_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件：{path}")

    lowered = path.lower()
    if lowered.endswith(".csv"):
        return pd.read_csv(path)
    if lowered.endswith(".xlsx") or lowered.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"暂不支持恢复该类型文件：{path}")


@app.post("/api/data/upload")
async def upload_file(
    file: UploadFile = File(...),
    project_name: str | None = Form(default=None),
):
    global current_df, original_df, current_filename, data_history, current_project_name

    if not file.filename or not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="当前仅支持 CSV 和 Excel 文件。")

    try:
        contents = await file.read()

        if file.filename.endswith(".csv"):
            current_df = pd.read_csv(io.BytesIO(contents))
        else:
            try:
                current_df = pd.read_excel(io.BytesIO(contents))
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="上传 Excel 文件需要在后端环境中安装 openpyxl。",
                ) from exc

        current_df.dropna(how="all", inplace=True)
        current_df.dropna(how="all", axis=1, inplace=True)

        if current_df.empty or current_df.shape[1] < 2:
            raise HTTPException(
                status_code=400,
                detail="数据集至少需要包含 1 个特征列和 1 个目标列。",
            )

        original_df = current_df.copy()
        current_filename = file.filename
        data_history = []
        current_project_name = _normalize_project_name(project_name) or current_project_name

        dataset_payload = serialize_dataset(current_df, current_filename, data_history)
        database_record = record_dataset_upload(
            current_project_name,
            file.filename,
            contents,
            dataset_payload,
        )
        return {
            **dataset_payload,
            "database_record": database_record,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/data/profile")
def get_data_profile():
    df = _require_current_df()
    return serialize_dataset(df, current_filename, data_history)


@app.post("/api/data/process")
def process_data(params: DataProcessParams):
    global current_df, data_history

    df = _require_current_df()
    if original_df is None:
        raise HTTPException(status_code=400, detail="原始数据集不存在，请重新上传数据。")

    try:
        next_df, next_history, operation = apply_data_process(
            df,
            original_df,
            params.model_dump(),
            data_history,
        )
        if next_df.empty or next_df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="处理后数据为空，或字段数量不足。请调整当前操作。")

        current_df = next_df
        data_history = next_history
        return {
            "status": "success",
            "message": operation["detail"],
            "operation": operation,
            "dataset": serialize_dataset(current_df, current_filename, data_history),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"数据处理失败：{str(exc)}") from exc


@app.post("/api/analysis/run")
def run_analysis(params: AnalysisParams):
    global current_project_name
    df = _require_current_df()

    try:
        result = run_statistical_analysis(df, params.model_dump())
        current_project_name = _normalize_project_name(params.project_name) or current_project_name
        database_record = save_analysis_run(
            current_project_name,
            _current_dataset_payload(),
            result,
        )
        return {
            "status": "success",
            **result,
            "database_record": database_record,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"统计分析失败：{str(exc)}") from exc


@app.post("/api/report/generate")
def create_report(params: ReportParams):
    global current_project_name
    try:
        current_project_name = _normalize_project_name(params.project_name) or current_project_name
        report_result = generate_report(params.model_dump())
        database_record = save_report_record(
            current_project_name,
            params.dataset,
            report_result,
        )
        return {
            "status": "success",
            **report_result,
            "database_record": database_record,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"生成报告失败：{str(exc)}") from exc


@app.post("/api/model/train")
def train_model(params: TrainParams):
    global current_df, current_project_name

    if current_df is None:
        raise HTTPException(status_code=400, detail="当前还没有载入数据集，请先上传数据。")

    if params.target_column not in current_df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"数据集中找不到目标列：{params.target_column}",
        )

    feature_columns = list(dict.fromkeys(params.feature_columns))
    if not feature_columns:
        raise HTTPException(status_code=400, detail="请至少选择 1 个特征列。")

    missing_features = [column for column in feature_columns if column not in current_df.columns]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"数据集中找不到这些特征列：{', '.join(missing_features)}",
        )

    if params.target_column in feature_columns:
        raise HTTPException(
            status_code=400,
            detail="目标列不能同时出现在特征列中。",
        )

    try:
        working_df = current_df[feature_columns + [params.target_column]].copy()
        rows_before_cleaning = len(working_df)
        working_df = working_df.dropna(subset=[params.target_column])

        if working_df.empty:
            raise HTTPException(
                status_code=400,
                detail="清理空值后，当前目标列没有可用于训练的数据。",
            )

        features = working_df[feature_columns]
        target = working_df[params.target_column]

        if params.task_type == "regression":
            numeric_target = pd.to_numeric(target, errors="coerce")
            valid_mask = numeric_target.notna()
            features = features.loc[valid_mask]
            target = numeric_target.loc[valid_mask]

            if len(target) < 4:
                raise HTTPException(
                    status_code=400,
                    detail="回归任务要求目标列至少有 4 行可用数据。",
                )
        else:
            if _looks_like_continuous_target(target):
                raise HTTPException(
                    status_code=400,
                    detail="当前目标列看起来是连续数值变量，不适合直接做分类。请切换到“回归”，或先把目标列分箱成有限类别后再做分类。",
                )

            target = target.astype(str).str.strip()
            valid_mask = target != ""
            features = features.loc[valid_mask]
            target = target.loc[valid_mask]

            if target.nunique() < 2:
                raise HTTPException(
                    status_code=400,
                    detail="分类任务要求目标列至少包含 2 个不同类别。",
                )

            if len(target) < 4:
                raise HTTPException(
                    status_code=400,
                    detail="分类任务要求目标列至少有 4 行可用数据。",
                )

            label_encoder = LabelEncoder()
            target = pd.Series(
                label_encoder.fit_transform(target),
                index=target.index,
                name=params.target_column,
            )

        stratify = None
        class_labels: list[str] | None = None

        if params.task_type == "classification":
            class_counts = target.value_counts()
            if class_counts.min() >= 2:
                stratify = target
            class_labels = [str(label) for label in label_encoder.classes_]

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocess", _build_preprocessor(features)),
                ("model", _build_model(params.task_type, params.model_type, params.hyperparameters)),
            ]
        )

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        if params.task_type == "classification":
            class_label_indexes = list(range(len(class_labels or [])))
            per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
                y_test,
                predictions,
                average=None,
                labels=class_label_indexes,
                zero_division=0,
            )

            metrics = {
                "accuracy": round(accuracy_score(y_test, predictions), 4),
                "precision": round(
                    precision_score(y_test, predictions, average="weighted", zero_division=0),
                    4,
                ),
                "recall": round(
                    recall_score(y_test, predictions, average="weighted", zero_division=0),
                    4,
                ),
                "f1_score": round(
                    f1_score(y_test, predictions, average="weighted", zero_division=0),
                    4,
                ),
                "type": "Classification",
            }

            confusion = confusion_matrix(y_test, predictions, labels=class_label_indexes)
            confusion_rows = [
                {
                    "actual": class_labels[row_index],
                    "values": [
                        {
                            "predicted": class_labels[column_index],
                            "count": int(confusion[row_index][column_index]),
                        }
                        for column_index in range(len(class_labels))
                    ],
                }
                for row_index in range(len(class_labels))
            ]

            class_performance = [
                {
                    "class_name": class_labels[index],
                    "precision": round(float(per_class_precision[index]) * 100, 2),
                    "recall": round(float(per_class_recall[index]) * 100, 2),
                    "f1_score": round(float(per_class_f1[index]) * 100, 2),
                    "support": int(per_class_support[index]),
                }
                for index in range(len(class_labels))
            ]
        else:
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            metrics = {
                "mse": round(mse, 4),
                "rmse": round(mse**0.5, 4),
                "mae": round(mae, 4),
                "r2_score": round(r2_score(y_test, predictions), 4),
                "type": "Regression",
            }

            regression_points = [
                {
                    "actual": round(float(actual), 4),
                    "predicted": round(float(predicted), 4),
                    "residual": round(float(predicted - actual), 4),
                }
                for actual, predicted in sorted(zip(y_test.tolist(), predictions.tolist()), key=lambda item: item[0])
            ]

        visualizations: dict[str, Any] = {
            "metric_bars": _build_metric_bars(params.task_type, metrics),
            "feature_effects": _build_feature_effects(pipeline),
        }

        if params.task_type == "classification":
            visualizations["confusion_matrix"] = confusion_rows
            visualizations["class_performance"] = class_performance
        else:
            visualizations["prediction_scatter"] = _compact_points(regression_points)

        current_project_name = _normalize_project_name(params.project_name) or current_project_name
        training_result = {
            "status": "success",
            "message": _get_success_message(params.model_type, params.task_type),
            "metrics": metrics,
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "task_type": params.task_type,
            "model_type": params.model_type,
            "model_label": _get_model_label(params.model_type, params.task_type),
            "target_column": params.target_column,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "hyperparameters": params.hyperparameters,
            "dropped_rows": rows_before_cleaning - len(features),
            "classes": class_labels,
            "backend_version": BACKEND_VERSION,
            "visualizations": visualizations,
        }
        database_record = save_training_run(
            current_project_name,
            _current_dataset_payload(),
            training_result,
        )
        return {
            **training_result,
            "database_record": database_record,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"训练失败：{str(exc)}") from exc


@app.post("/api/project/save")
def save_project(params: SnapshotParams):
    global current_project_name, current_project_notes

    project_name = (params.project_name or "").strip()
    if not project_name:
        raise HTTPException(status_code=400, detail="项目名称不能为空。")

    snapshot_dataset_path = None
    if current_df is not None:
        snapshot_dataset_path = save_dataframe_snapshot(
            project_name,
            current_df,
            current_filename,
        )

    snapshot_payload = {
        "project_name": project_name,
        "project_notes": params.project_notes or "",
        "saved_at": _get_timestamp_iso(),
        "dataset": params.dataset,
        "dataset_storage_path": snapshot_dataset_path,
        "dataset_storage_filename": current_filename,
        "processing_history": params.processing_history,
        "training_result": params.training_result,
        "analysis_results": params.analysis_results,
        "report_result": params.report_result,
        "workspace_state": params.workspace_state,
    }

    try:
        current_project_name = project_name
        current_project_notes = params.project_notes or ""
        database_record = save_project_snapshot(
            project_name,
            current_project_notes,
            snapshot_payload,
        )
        return {
            "status": "success",
            "message": "项目快照已保存。",
            "snapshot_id": database_record["snapshot_id"],
            "project": database_record["project"],
            "database": get_database_info(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存项目快照失败：{str(exc)}") from exc


@app.get("/api/projects")
def get_projects(limit: int = 12):
    try:
        return {
            "status": "success",
            "items": list_recent_projects(limit),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取项目列表失败：{str(exc)}") from exc


@app.get("/api/projects/{project_id}")
def open_project(project_id: int):
    global current_df, original_df, current_filename, data_history, current_project_name, current_project_notes

    detail = get_project_detail(project_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="找不到对应的项目记录。")

    project = detail["project"]
    snapshot = detail.get("latest_snapshot") or {}
    snapshot_data = snapshot.get("data") or {}
    latest_upload = detail.get("latest_upload") or {}

    restore_path = snapshot_data.get("dataset_storage_path") or latest_upload.get("stored_path") or project.get("latest_dataset_path")
    restored_dataset = None

    if restore_path:
        try:
            restored_df = _load_dataframe_from_path(restore_path)
            restored_df.dropna(how="all", inplace=True)
            restored_df.dropna(how="all", axis=1, inplace=True)
            if restored_df.empty or restored_df.shape[1] < 2:
                raise ValueError("恢复后的数据集为空，或字段数量不足。")

            current_df = restored_df
            original_df = restored_df.copy()
            current_filename = (
                snapshot_data.get("dataset_storage_filename")
                or snapshot_data.get("dataset", {}).get("filename")
                or project.get("latest_dataset_filename")
            )
            data_history = snapshot_data.get("processing_history") or []
            restored_dataset = serialize_dataset(current_df, current_filename, data_history)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"恢复项目数据失败：{str(exc)}") from exc

    current_project_name = project["name"]
    current_project_notes = snapshot_data.get("project_notes") or project.get("notes") or ""

    return {
        "status": "success",
        "project": project,
        "snapshot": snapshot,
        "dataset": restored_dataset,
        "latest_upload": latest_upload,
    }


def _get_timestamp_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    import sys
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=DEFAULT_PORT,
        reload=not getattr(sys, "frozen", False),
    )
