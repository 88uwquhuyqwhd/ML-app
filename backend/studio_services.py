from __future__ import annotations

from datetime import datetime
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


def _round_number(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric) or np.isinf(numeric):
        return None
    return round(numeric, digits)


def _compact_points(points: list[dict[str, Any]], max_points: int = 80) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    return [points[int(index * step)] for index in range(max_points)]


def _stringify(value: Any) -> str:
    if pd.isna(value):
        return "NaN"
    return str(value)


def _build_table(title: str, columns: list[dict[str, str]], rows: list[dict[str, Any]]):
    return {
        "title": title,
        "columns": columns,
        "rows": rows,
    }


def _build_bar_visual(
    title: str,
    description: str,
    data: list[dict[str, Any]],
    x_key: str,
    series: list[dict[str, str]],
    layout: str = "vertical",
):
    return {
        "type": "bar",
        "title": title,
        "description": description,
        "data": data,
        "x_key": x_key,
        "series": series,
        "layout": layout,
    }


def _build_matrix_visual(
    title: str,
    description: str,
    rows: list[str],
    columns: list[str],
    cells: list[dict[str, Any]],
):
    return {
        "type": "matrix",
        "title": title,
        "description": description,
        "rows": rows,
        "columns": columns,
        "cells": cells,
    }


def _build_scatter_visual(
    title: str,
    description: str,
    points: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    series_key: str | None = None,
):
    return {
        "type": "scatter",
        "title": title,
        "description": description,
        "points": points,
        "x_key": x_key,
        "y_key": y_key,
        "x_label": x_label,
        "y_label": y_label,
        "series_key": series_key,
    }


def _get_preview_row_limit() -> int:
    raw_value = str(os.getenv("ML_APP_PREVIEW_LIMIT", "200")).strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        parsed = 200
    return min(max(parsed, 20), 1000)


def _serialize_preview(df: pd.DataFrame) -> tuple[list[dict[str, Any]], int, bool]:
    limit = _get_preview_row_limit()
    preview_df = df.head(limit).copy()
    preview_df = preview_df.fillna("NaN")
    preview_rows = preview_df.to_dict(orient="records")
    return preview_rows, len(preview_rows), len(df) > len(preview_rows)


def serialize_dataset(df: pd.DataFrame, filename: str | None, history: list[dict[str, Any]] | None = None):
    preview_rows, preview_row_count, preview_truncated = _serialize_preview(df)
    missing_series = df.isna().sum().sort_values(ascending=False)
    missing_summary = [
        {
            "column": column,
            "missing": int(count),
            "missing_rate": round((count / max(len(df), 1)) * 100, 2),
        }
        for column, count in missing_series.items()
        if count > 0
    ][:16]

    column_profiles = []
    for column in df.columns:
        series = df[column]
        sample_values = [_stringify(value) for value in series.dropna().head(3).tolist()]
        column_profiles.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "sample_values": sample_values,
            }
        )

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]

    numeric_summary = []
    if numeric_columns:
        described = df[numeric_columns].describe().transpose().reset_index().rename(columns={"index": "column"})
        for row in described.to_dict(orient="records"):
            numeric_summary.append(
                {
                    "column": row["column"],
                    "mean": _round_number(row.get("mean")),
                    "std": _round_number(row.get("std")),
                    "min": _round_number(row.get("min")),
                    "max": _round_number(row.get("max")),
                }
            )

    return {
        "filename": filename or "当前数据集",
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "preview": preview_rows,
        "preview_row_count": preview_row_count,
        "preview_total_rows": int(len(df)),
        "preview_truncated": preview_truncated,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_summary": missing_summary,
        "column_profiles": column_profiles,
        "numeric_summary": numeric_summary[:16],
        "history": history or [],
    }


def apply_data_process(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    params: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
):
    action = params.get("action")
    columns = [column for column in params.get("columns", []) if column in df.columns]
    next_df = df.copy()
    detail = ""
    affected_rows = 0
    created_columns: list[str] = []

    if action == "reset_data":
        next_df = original_df.copy()
        detail = "已恢复到原始上传数据。"
    elif action == "handle_missing":
        strategy = params.get("strategy", "drop_rows")
        target_columns = columns or next_df.columns.tolist()
        if strategy == "drop_rows":
            before = len(next_df)
            next_df = next_df.dropna(subset=target_columns)
            affected_rows = before - len(next_df)
            detail = f"按 {len(target_columns)} 个字段删除缺失行。"
        elif strategy in {"fill_mean", "fill_median"}:
            numeric_columns = next_df[target_columns].select_dtypes(include=["number"]).columns.tolist()
            if not numeric_columns:
                raise ValueError("该策略只适用于数值列，请先选择数值字段。")
            for column in numeric_columns:
                filler = next_df[column].mean() if strategy == "fill_mean" else next_df[column].median()
                next_df[column] = next_df[column].fillna(filler)
            detail = "已使用数值统计量填补缺失值。"
        elif strategy == "fill_mode":
            for column in target_columns:
                mode = next_df[column].mode(dropna=True)
                if not mode.empty:
                    next_df[column] = next_df[column].fillna(mode.iloc[0])
            detail = "已使用众数填补缺失值。"
        elif strategy == "fill_value":
            fill_value = params.get("fill_value", "")
            next_df[target_columns] = next_df[target_columns].fillna(fill_value)
            detail = f"已用固定值 {fill_value} 填补缺失。"
        else:
            raise ValueError("不支持的缺失值处理方式。")
    elif action == "handle_outlier":
        method = params.get("method", "clip_iqr")
        threshold = float(params.get("threshold", 1.5) or 1.5)
        numeric_columns = next_df[columns].select_dtypes(include=["number"]).columns.tolist()
        if not numeric_columns:
            raise ValueError("异常值处理至少需要 1 个数值字段。")
        if method in {"clip_iqr", "remove_iqr"}:
            mask = pd.Series(True, index=next_df.index)
            for column in numeric_columns:
                q1 = next_df[column].quantile(0.25)
                q3 = next_df[column].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                if method == "clip_iqr":
                    next_df[column] = next_df[column].clip(lower=lower, upper=upper)
                else:
                    column_mask = next_df[column].isna() | next_df[column].between(lower, upper)
                    mask &= column_mask
            if method == "remove_iqr":
                before = len(next_df)
                next_df = next_df.loc[mask].copy()
                affected_rows = before - len(next_df)
            detail = "已按 IQR 方法处理异常值。"
        else:
            raise ValueError("当前仅支持 IQR 异常值处理。")
    elif action == "encode_columns":
        method = params.get("method", "label")
        if not columns:
            raise ValueError("请至少选择 1 个要编码的字段。")
        if method == "label":
            for column in columns:
                next_df[column] = pd.factorize(next_df[column].astype(str).fillna("NaN"))[0]
            detail = "已将所选分类字段编码为整数标签。"
        elif method == "one_hot":
            next_df = pd.get_dummies(next_df, columns=columns, dummy_na=False)
            created_columns = [column for column in next_df.columns if column not in df.columns]
            detail = "已将所选分类字段展开为哑变量列。"
        else:
            raise ValueError("不支持的编码方式。")
    elif action == "scale_columns":
        method = params.get("method", "standardize")
        numeric_columns = next_df[columns].select_dtypes(include=["number"]).columns.tolist()
        if not numeric_columns:
            raise ValueError("标准化至少需要 1 个数值字段。")
        scaler = StandardScaler() if method == "standardize" else MinMaxScaler()
        next_df[numeric_columns] = scaler.fit_transform(next_df[numeric_columns])
        detail = "已完成数值字段缩放。"
    elif action == "cast_columns":
        target_dtype = params.get("target_dtype", "string")
        if not columns:
            raise ValueError("请至少选择 1 个字段用于类型转换。")
        for column in columns:
            if target_dtype == "number":
                next_df[column] = pd.to_numeric(next_df[column], errors="coerce")
            elif target_dtype == "string":
                next_df[column] = next_df[column].astype(str)
            elif target_dtype == "category":
                next_df[column] = next_df[column].astype("category")
            elif target_dtype == "datetime":
                next_df[column] = pd.to_datetime(next_df[column], errors="coerce")
            else:
                raise ValueError("不支持的目标字段类型。")
        detail = "已完成字段类型转换。"
    elif action == "filter_rows":
        expression = str(params.get("filter_expression", "")).strip()
        if not expression:
            raise ValueError("请输入筛选表达式。")
        before = len(next_df)
        next_df = next_df.query(expression, engine="python").copy()
        affected_rows = before - len(next_df)
        detail = f"已按表达式筛选数据：{expression}"
    elif action == "derive_column":
        new_column = str(params.get("new_column", "")).strip()
        expression = str(params.get("expression", "")).strip()
        if not new_column or not expression:
            raise ValueError("请填写新字段名称和计算表达式。")
        next_df[new_column] = next_df.eval(expression)
        created_columns = [new_column]
        detail = f"已创建派生字段 {new_column}。"
    else:
        raise ValueError("不支持的数据处理操作。")

    operation = {
        "action": action,
        "label": {
            "handle_missing": "缺失值处理",
            "handle_outlier": "异常值处理",
            "encode_columns": "编码转换",
            "scale_columns": "标准化",
            "cast_columns": "字段类型转换",
            "filter_rows": "筛选行",
            "derive_column": "派生变量",
            "reset_data": "重置数据",
        }.get(action, action),
        "detail": detail,
        "affected_rows": int(affected_rows),
        "created_columns": created_columns,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    next_history = [*(history or []), operation] if action != "reset_data" else [operation]
    return next_df, next_history, operation


def _preprocess_for_modeling(df: pd.DataFrame, features: list[str], target_column: str):
    working_df = df[features + [target_column]].dropna(subset=[target_column]).copy()
    x = working_df[features]
    y = working_df[target_column]
    numeric_columns = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in x.columns if column not in numeric_columns]

    transformers = []
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

    preprocessor = ColumnTransformer(transformers=transformers)
    return working_df, x, y, preprocessor


def run_statistical_analysis(df: pd.DataFrame, params: dict[str, Any]):
    analysis_type = params.get("analysis_type")
    title = ""
    summary = ""
    insights: list[str] = []
    tables: list[dict[str, Any]] = []
    visualizations: list[dict[str, Any]] = []

    if analysis_type == "descriptive":
        columns = params.get("columns") or df.select_dtypes(include=["number"]).columns.tolist()
        if not columns:
            raise ValueError("描述统计至少需要 1 个数值字段。")
        summary_df = df[columns].describe().transpose().reset_index().rename(columns={"index": "column"})
        table_rows = [
            {
                "column": row["column"],
                "count": int(row["count"]),
                "mean": _round_number(row.get("mean")),
                "std": _round_number(row.get("std")),
                "min": _round_number(row.get("min")),
                "max": _round_number(row.get("max")),
            }
            for row in summary_df.to_dict(orient="records")
        ]
        title = "描述统计"
        summary = f"已对 {len(columns)} 个数值字段完成描述统计，适合先快速了解均值、波动和范围。"
        insights = [f"字段 {row['column']} 的均值为 {row['mean']}，标准差为 {row['std']}。" for row in table_rows[:3]]
        tables.append(
            _build_table(
                "描述统计结果",
                [
                    {"key": "column", "label": "字段"},
                    {"key": "count", "label": "样本数"},
                    {"key": "mean", "label": "均值"},
                    {"key": "std", "label": "标准差"},
                    {"key": "min", "label": "最小值"},
                    {"key": "max", "label": "最大值"},
                ],
                table_rows,
            )
        )
        visualizations.append(
            _build_bar_visual(
                "均值对比",
                "用柱图快速比较各字段的均值水平。",
                [{"column": row["column"], "mean": row["mean"] or 0} for row in table_rows],
                "column",
                [{"key": "mean", "label": "均值", "color": "#d97757"}],
            )
        )
    elif analysis_type == "correlation":
        columns = params.get("columns") or df.select_dtypes(include=["number"]).columns.tolist()
        if len(columns) < 2:
            raise ValueError("相关分析至少需要 2 个数值字段。")
        corr = df[columns].corr().round(4)
        cells = [
            {
                "row": row,
                "column": column,
                "value": _round_number(corr.loc[row, column], 4),
                "label": str(_round_number(corr.loc[row, column], 4)),
            }
            for row in columns
            for column in columns
        ]
        title = "相关分析"
        summary = f"已计算 {len(columns)} 个字段的相关系数矩阵。"
        insights = [
            "建议重点关注绝对值较高的相关系数，识别强相关关系和潜在共线性。",
            "对角线恒为 1，非对角线越接近 1 或 -1，线性关系越强。",
        ]
        visualizations.append(
            _build_matrix_visual(
                "相关系数矩阵",
                "颜色和数值共同反映字段之间的线性相关程度。",
                columns,
                columns,
                cells,
            )
        )
        tables.append(
            _build_table(
                "相关系数表",
                [{"key": "row", "label": "字段"}] + [{"key": column, "label": column} for column in columns],
                [{"row": row, **{column: _round_number(corr.loc[row, column], 4) for column in columns}} for row in columns],
            )
        )
    elif analysis_type in {"ttest", "anova"}:
        target_column = params.get("target_column")
        group_column = params.get("group_column")
        if target_column not in df.columns or group_column not in df.columns:
            raise ValueError("请正确选择数值目标列和分组列。")
        working = df[[target_column, group_column]].dropna().copy()
        groups = [(str(name), values[target_column].astype(float).tolist()) for name, values in working.groupby(group_column)]
        if len(groups) < 2:
            raise ValueError("分组列至少需要 2 个有效类别。")
        group_means = [{"group": name, "mean": _round_number(np.mean(values)), "count": len(values)} for name, values in groups]
        visualizations.append(
            _build_bar_visual(
                "分组均值",
                "用来比较不同组之间的平均水平差异。",
                group_means,
                "group",
                [{"key": "mean", "label": "均值", "color": "#6a9bcc"}],
            )
        )
        if analysis_type == "ttest":
            if len(groups) != 2:
                raise ValueError("T 检验要求分组列恰好包含 2 个类别。")
            statistic, p_value = stats.ttest_ind(groups[0][1], groups[1][1], equal_var=False, nan_policy="omit")
            title = "T 检验"
            summary = f"已比较 {groups[0][0]} 与 {groups[1][0]} 在 {target_column} 上的均值差异。"
            insights = [
                f"t 值为 {_round_number(statistic, 4)}，p 值为 {_round_number(p_value, 4)}。",
                "通常当 p < 0.05 时，可认为两组均值差异具有统计意义。",
            ]
            tables.append(
                _build_table(
                    "T 检验结果",
                    [
                        {"key": "metric", "label": "指标"},
                        {"key": "value", "label": "结果"},
                    ],
                    [
                        {"metric": "组别 1", "value": groups[0][0]},
                        {"metric": "组别 2", "value": groups[1][0]},
                        {"metric": "t 值", "value": _round_number(statistic, 4)},
                        {"metric": "p 值", "value": _round_number(p_value, 4)},
                    ],
                )
            )
        else:
            statistic, p_value = stats.f_oneway(*[values for _, values in groups])
            title = "方差分析"
            summary = f"已比较 {len(groups)} 个组别在 {target_column} 上的均值差异。"
            insights = [
                f"F 值为 {_round_number(statistic, 4)}，p 值为 {_round_number(p_value, 4)}。",
                "若 p < 0.05，说明至少有一组的均值与其他组存在显著差异。",
            ]
            tables.append(
                _build_table(
                    "ANOVA 结果",
                    [{"key": "metric", "label": "指标"}, {"key": "value", "label": "结果"}],
                    [
                        {"metric": "分组字段", "value": group_column},
                        {"metric": "目标字段", "value": target_column},
                        {"metric": "F 值", "value": _round_number(statistic, 4)},
                        {"metric": "p 值", "value": _round_number(p_value, 4)},
                    ],
                )
            )
    elif analysis_type == "chi_square":
        row_column = params.get("row_column")
        column_column = params.get("column_column")
        if row_column not in df.columns or column_column not in df.columns:
            raise ValueError("请正确选择两个分类字段。")
        working = df[[row_column, column_column]].dropna()
        contingency = pd.crosstab(working[row_column], working[column_column])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        rows = [str(index) for index in contingency.index.tolist()]
        columns = [str(column) for column in contingency.columns.tolist()]
        cells = [
            {
                "row": str(row),
                "column": str(column),
                "value": int(contingency.loc[row, column]),
                "label": str(int(contingency.loc[row, column])),
            }
            for row in contingency.index
            for column in contingency.columns
        ]
        title = "卡方检验"
        summary = f"已分析 {row_column} 与 {column_column} 的列联关系。"
        insights = [
            f"卡方统计量为 {_round_number(chi2, 4)}，p 值为 {_round_number(p_value, 4)}。",
            "若 p < 0.05，可认为两个分类变量之间存在统计关联。",
        ]
        visualizations.append(
            _build_matrix_visual(
                "列联表热力图",
                "颜色越深表示该组合下的样本数越多。",
                rows,
                columns,
                cells,
            )
        )
        tables.append(
            _build_table(
                "卡方检验结果",
                [{"key": "metric", "label": "指标"}, {"key": "value", "label": "结果"}],
                [
                    {"metric": "行变量", "value": row_column},
                    {"metric": "列变量", "value": column_column},
                    {"metric": "卡方值", "value": _round_number(chi2, 4)},
                    {"metric": "p 值", "value": _round_number(p_value, 4)},
                ],
            )
        )
    elif analysis_type in {"linear_regression", "logistic_regression"}:
        target_column = params.get("target_column")
        feature_columns = [column for column in params.get("feature_columns", []) if column in df.columns and column != target_column]
        if target_column not in df.columns or not feature_columns:
            raise ValueError("请正确选择目标列和特征列。")
        working, x, y, preprocessor = _preprocess_for_modeling(df, feature_columns, target_column)
        if len(working) < 6:
            raise ValueError("样本量过少，至少需要 6 行有效数据。")

        if analysis_type == "linear_regression":
            working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
            working = working.dropna(subset=[target_column])
            x = working[feature_columns]
            y = working[target_column]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            title = "线性回归"
        else:
            working[target_column] = working[target_column].astype(str).str.strip()
            working = working.loc[working[target_column] != ""].copy()
            x = working[feature_columns]
            y = working[target_column]
            encoder = LabelEncoder()
            encoded_y = encoder.fit_transform(y)
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                encoded_y,
                test_size=0.2,
                random_state=42,
                stratify=encoded_y if len(np.unique(encoded_y)) > 1 else None,
            )
            model = LogisticRegression(max_iter=1000)
            title = "Logit 回归"

        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        coefficients = pipeline.named_steps["model"].coef_
        if getattr(coefficients, "ndim", 1) > 1:
            coefficient_values = np.abs(coefficients).mean(axis=0)
        else:
            coefficient_values = np.abs(coefficients)
        effect_rows = [
            {"feature": str(name).replace("numeric__", "").replace("categorical__", ""), "value": _round_number(value, 6)}
            for name, value in zip(feature_names, coefficient_values)
        ]
        effect_rows = sorted(effect_rows, key=lambda item: item["value"] or 0, reverse=True)[:12]
        visualizations.append(
            _build_bar_visual(
                "特征影响",
                "系数绝对值越大，表示该特征对模型输出的影响越强。",
                effect_rows,
                "feature",
                [{"key": "value", "label": "影响值", "color": "#6a9bcc"}],
                layout="horizontal",
            )
        )

        if analysis_type == "linear_regression":
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            scatter_points = _compact_points(
                [
                    {"actual": _round_number(actual), "predicted": _round_number(predicted)}
                    for actual, predicted in zip(y_test.tolist(), predictions.tolist())
                ]
            )
            summary = f"已完成 {target_column} 的线性回归分析，测试集 R2 为 {_round_number(r2, 4)}。"
            insights = [
                f"RMSE 为 {_round_number(mse**0.5, 4)}，可用来观察预测误差量级。",
                "如果散点更贴近对角线，说明预测值和实际值更接近。",
            ]
            visualizations.append(
                _build_scatter_visual(
                    "预测散点",
                    "比较测试集中的实际值与预测值。",
                    scatter_points,
                    "actual",
                    "predicted",
                    "实际值",
                    "预测值",
                )
            )
            tables.append(
                _build_table(
                    "线性回归指标",
                    [{"key": "metric", "label": "指标"}, {"key": "value", "label": "结果"}],
                    [
                        {"metric": "R2", "value": _round_number(r2, 4)},
                        {"metric": "RMSE", "value": _round_number(mse**0.5, 4)},
                        {"metric": "特征数", "value": len(feature_columns)},
                    ],
                )
            )
        else:
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
            recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
            labels = [str(label) for label in encoder.classes_]
            cm = confusion_matrix(y_test, predictions)
            cells = [
                {
                    "row": labels[row_index],
                    "column": labels[column_index],
                    "value": int(cm[row_index][column_index]),
                    "label": str(int(cm[row_index][column_index])),
                }
                for row_index in range(len(labels))
                for column_index in range(len(labels))
            ]
            summary = f"已完成 {target_column} 的 Logit 回归分析，测试集准确率为 {_round_number(accuracy * 100, 2)}%。"
            insights = [
                f"加权精确率为 {_round_number(precision * 100, 2)}%，召回率为 {_round_number(recall * 100, 2)}%。",
                "混淆矩阵适合继续查看具体是哪些类别最容易混淆。",
            ]
            visualizations.append(
                _build_matrix_visual(
                    "混淆矩阵",
                    "查看分类结果在不同类别上的混淆情况。",
                    labels,
                    labels,
                    cells,
                )
            )
            tables.append(
                _build_table(
                    "Logit 回归指标",
                    [{"key": "metric", "label": "指标"}, {"key": "value", "label": "结果"}],
                    [
                        {"metric": "准确率", "value": _round_number(accuracy * 100, 2)},
                        {"metric": "精确率", "value": _round_number(precision * 100, 2)},
                        {"metric": "召回率", "value": _round_number(recall * 100, 2)},
                    ],
                )
            )
    elif analysis_type == "clustering":
        columns = params.get("columns") or []
        feature_df = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
        if feature_df.shape[1] < 2:
            raise ValueError("聚类分析至少需要 2 个数值字段。")
        n_clusters = int(params.get("n_clusters", 3) or 3)
        scaled = StandardScaler().fit_transform(feature_df)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        title = "聚类分析"
        summary = f"已按 {feature_df.shape[1]} 个字段将样本划分为 {n_clusters} 个簇。"
        insights = ["建议结合簇规模和聚类投影，判断样本是否形成较清晰的分组。"]
        visualizations.append(
            _build_bar_visual(
                "簇规模分布",
                "查看每个聚类中包含的样本数量。",
                [{"cluster": f"簇 {index + 1}", "size": int(size)} for index, size in cluster_sizes.items()],
                "cluster",
                [{"key": "size", "label": "样本数", "color": "#d97757"}],
            )
        )
        projection = PCA(n_components=2, random_state=42).fit_transform(scaled)
        scatter_points = _compact_points(
            [
                {
                    "x": _round_number(point[0]),
                    "y": _round_number(point[1]),
                    "cluster": f"簇 {labels[index] + 1}",
                }
                for index, point in enumerate(projection)
            ]
        )
        visualizations.append(
            _build_scatter_visual(
                "聚类投影",
                "用 PCA 将高维特征压缩到二维后查看簇之间的分布。",
                scatter_points,
                "x",
                "y",
                "主成分 1",
                "主成分 2",
                "cluster",
            )
        )
    elif analysis_type == "pca":
        columns = params.get("columns") or []
        feature_df = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
        if feature_df.shape[1] < 2:
            raise ValueError("PCA 至少需要 2 个数值字段。")
        n_components = min(int(params.get("n_components", 2) or 2), feature_df.shape[1])
        scaled = StandardScaler().fit_transform(feature_df)
        model = PCA(n_components=n_components, random_state=42)
        model.fit(scaled)
        explained = model.explained_variance_ratio_
        title = "PCA 主成分分析"
        summary = f"已提取 {n_components} 个主成分，累计解释率为 {_round_number(float(explained.sum()) * 100, 2)}%。"
        insights = ["主成分解释率越高，表示它能保留原始变量更多的信息。"]
        visualizations.append(
            _build_bar_visual(
                "解释率",
                "用于判断各主成分保留信息的能力。",
                [{"component": f"PC{i + 1}", "ratio": _round_number(float(value) * 100, 2)} for i, value in enumerate(explained)],
                "component",
                [{"key": "ratio", "label": "解释率(%)", "color": "#788c5d"}],
            )
        )
        loading_rows = []
        for feature_index, feature in enumerate(feature_df.columns):
            row = {"feature": feature}
            for component_index in range(n_components):
                row[f"PC{component_index + 1}"] = _round_number(model.components_[component_index][feature_index], 4)
            loading_rows.append(row)
        tables.append(
            _build_table(
                "PCA 载荷矩阵",
                [{"key": "feature", "label": "字段"}] + [{"key": f"PC{i + 1}", "label": f"PC{i + 1}"} for i in range(n_components)],
                loading_rows,
            )
        )
    elif analysis_type == "factor_analysis":
        columns = params.get("columns") or []
        feature_df = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
        if feature_df.shape[1] < 2:
            raise ValueError("因子分析至少需要 2 个数值字段。")
        n_factors = min(int(params.get("n_factors", 2) or 2), feature_df.shape[1])
        scaled = StandardScaler().fit_transform(feature_df)
        model = FactorAnalysis(n_components=n_factors, random_state=42)
        model.fit(scaled)
        loadings = model.components_.transpose()
        cells = []
        factor_labels = [f"因子 {index + 1}" for index in range(n_factors)]
        for row_index, feature in enumerate(feature_df.columns):
            for col_index, factor in enumerate(factor_labels):
                value = _round_number(loadings[row_index][col_index], 4)
                cells.append(
                    {
                        "row": feature,
                        "column": factor,
                        "value": value,
                        "label": str(value),
                    }
                )
        communalities = np.sum(loadings**2, axis=1)
        title = "因子分析"
        summary = f"已提取 {n_factors} 个公共因子，可用于观察变量背后的潜在结构。"
        insights = ["如果某个变量在某个因子上的载荷更高，说明该变量与该潜在因子关系更强。"]
        visualizations.append(
            _build_matrix_visual(
                "因子载荷热力图",
                "查看每个变量在不同因子上的载荷强弱。",
                feature_df.columns.tolist(),
                factor_labels,
                cells,
            )
        )
        tables.append(
            _build_table(
                "共同度",
                [{"key": "feature", "label": "字段"}, {"key": "communality", "label": "共同度"}],
                [{"feature": feature_df.columns[index], "communality": _round_number(value, 4)} for index, value in enumerate(communalities)],
            )
        )
    else:
        raise ValueError("不支持的统计分析类型。")

    return {
        "analysis_type": analysis_type,
        "title": title,
        "summary": summary,
        "insights": insights,
        "tables": tables,
        "visualizations": visualizations,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_report(payload: dict[str, Any]):
    project_name = payload.get("project_name") or "未命名项目"
    dataset = payload.get("dataset") or {}
    processing_history = payload.get("processing_history") or []
    training_result = payload.get("training_result") or {}
    analysis_results = payload.get("analysis_results") or []

    sections = [
        {
            "title": "项目概览",
            "body": f"项目“{project_name}”当前使用数据集 {dataset.get('filename', '当前数据集')}，"
            f"共有 {dataset.get('shape', [0, 0])[0]} 行、{dataset.get('shape', [0, 0])[1]} 列。",
            "bullets": [
                f"数值字段 {len(dataset.get('numeric_columns', []))} 个",
                f"分类字段 {len(dataset.get('categorical_columns', []))} 个",
            ],
        }
    ]

    if processing_history:
        sections.append(
            {
                "title": "数据准备",
                "body": f"共执行了 {len(processing_history)} 步数据处理，最近一次操作为 {processing_history[-1].get('label', '数据处理')}。",
                "bullets": [f"{item.get('label', '')}：{item.get('detail', '')}" for item in processing_history[-5:]],
            }
        )

    if training_result:
        metrics = training_result.get("metrics", {})
        metric_text = (
            f"准确率 {round(metrics.get('accuracy', 0) * 100, 2)}%"
            if training_result.get("task_type") == "classification"
            else f"R2 {metrics.get('r2_score', '-')}, RMSE {metrics.get('rmse', '-')}"
        )
        sections.append(
            {
                "title": "建模结果",
                "body": f"当前模型为 {training_result.get('model_label', training_result.get('model_type', '模型'))}，{training_result.get('message', '')}。",
                "bullets": [
                    f"目标列：{training_result.get('target_column', '-')}",
                    f"特征数：{training_result.get('feature_count', '-')}",
                    f"核心表现：{metric_text}",
                ],
            }
        )

    if analysis_results:
        sections.append(
            {
                "title": "统计分析",
                "body": f"本次报告共纳入 {len(analysis_results)} 份统计分析结果。",
                "bullets": [f"{item.get('title', '分析')}：{item.get('summary', '')}" for item in analysis_results[-6:]],
            }
        )

    sections.append(
        {
            "title": "下一步建议",
            "body": "建议先根据数据准备和统计分析结果确认字段质量，再对比不同模型配置并沉淀标准分析模板。",
            "bullets": [
                "优先复核缺失值和异常值处理是否符合业务规则",
                "将高价值统计分析结果纳入项目模板，便于复用",
                "把当前报告导出为 Word 或打印为 PDF，用于汇报或存档",
            ],
        }
    )

    markdown_parts = [f"# {project_name} 分析报告", ""]
    html_parts = [
        "<html><head><meta charset='utf-8'><title>分析报告</title>",
        "<style>body{font-family:'Microsoft YaHei',sans-serif;padding:32px;color:#141413;}h1,h2{color:#d97757;}li{margin:6px 0;}section{margin-bottom:28px;}p{line-height:1.8;}</style>",
        "</head><body>",
        f"<h1>{project_name} 分析报告</h1>",
    ]

    for section in sections:
        markdown_parts.append(f"## {section['title']}")
        markdown_parts.append(section["body"])
        markdown_parts.extend([f"- {item}" for item in section["bullets"]])
        markdown_parts.append("")

        html_parts.append("<section>")
        html_parts.append(f"<h2>{section['title']}</h2>")
        html_parts.append(f"<p>{section['body']}</p><ul>")
        html_parts.extend([f"<li>{item}</li>" for item in section["bullets"]])
        html_parts.append("</ul></section>")

    html_parts.append("</body></html>")

    return {
        "title": f"{project_name} 分析报告",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sections": sections,
        "markdown": "\n".join(markdown_parts),
        "html": "".join(html_parts),
    }
