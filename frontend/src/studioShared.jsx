import { Box, Chip, Paper, Stack, Typography } from '@mui/material'

export const API_BASE = import.meta.env?.VITE_API_BASE_URL || 'http://127.0.0.1:8010'
export const DEFAULT_MODEL_TYPE = 'RandomForest'
export const OLD_BACKEND_ERROR = 'Only classification is supported in this basic version.'
export const RADII = {
  hero: '18px',
  panel: '14px',
  card: '14px',
  inset: '10px',
}

export const TASK_LABELS = {
  classification: '分类',
  regression: '回归',
}

export const MODEL_LABELS = {
  RandomForest: '随机森林',
  SVM: '支持向量机',
  LogisticRegression: {
    classification: '逻辑回归',
    regression: '线性回归',
  },
}

export const DEFAULT_HYPERPARAMETERS = {
  classification: {
    RandomForest: { n_estimators: '100', max_depth: '' },
    SVM: { C: '1.0', kernel: 'rbf', gamma: 'scale' },
    LogisticRegression: { C: '1.0', max_iter: '1000' },
  },
  regression: {
    RandomForest: { n_estimators: '100', max_depth: '' },
    SVM: { C: '1.0', kernel: 'rbf', gamma: 'scale' },
    LogisticRegression: {},
  },
}

export function getDefaultHyperparameters(taskType, modelType) {
  return { ...(DEFAULT_HYPERPARAMETERS[taskType]?.[modelType] ?? {}) }
}

export function getModelLabel(modelType, taskType) {
  const modelLabel = MODEL_LABELS[modelType]
  if (typeof modelLabel === 'string') {
    return modelLabel
  }
  return modelLabel?.[taskType] ?? modelType
}

export function inferTaskType(dataInfo, targetColumn) {
  const dtype = String(dataInfo?.dtypes?.[targetColumn] ?? '').toLowerCase()
  const isNumericTarget =
    dtype.includes('int') ||
    dtype.includes('float') ||
    dtype.includes('double') ||
    dtype.includes('decimal')

  return isNumericTarget ? 'regression' : 'classification'
}

export function getAvailableFeatureColumns(dataInfo, targetColumn) {
  if (!dataInfo) {
    return []
  }
  const columns = Array.isArray(dataInfo.columns) ? dataInfo.columns : []
  return columns.filter((column) => column !== targetColumn)
}

export function formatErrorMessage(detail) {
  if (!detail) {
    return '请求失败，请稍后再试。'
  }

  if (detail.includes(OLD_BACKEND_ERROR) || detail.includes('Target column appears to be continuous')) {
    return '当前连接到的是旧版后端，它还不支持回归任务。请完全关闭桌面应用和开发终端后重新启动，再重新训练。'
  }

  if (detail.startsWith('Training failed:')) {
    return detail.replace('Training failed:', '训练失败：')
  }

  return detail
}

export function renderSelectedFeatures(selected) {
  if (!selected.length) {
    return '暂未选择'
  }

  if (selected.length <= 3) {
    return selected.join('，')
  }

  return `${selected.slice(0, 3).join('，')} 等 ${selected.length} 个字段`
}

export function formatNumber(value, digits = 4) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    return '-'
  }
  return parsed.toFixed(Math.abs(parsed) >= 100 ? Math.min(digits, 2) : digits)
}

export function formatPercent(value) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    return '-'
  }
  return `${parsed.toFixed(2)}%`
}

function isNumericDtype(dtype) {
  const lowered = String(dtype ?? '').toLowerCase()
  return (
    lowered.includes('int') ||
    lowered.includes('float') ||
    lowered.includes('double') ||
    lowered.includes('decimal')
  )
}

function toArray(value) {
  return Array.isArray(value) ? value : []
}

function toObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {}
}

function toDisplayString(value, fallback = '-') {
  if (value === undefined || value === null || value === '') {
    return fallback
  }
  return String(value)
}

export function getSafeShape(dataInfo) {
  const shape = toArray(dataInfo?.shape)
  const rowCount = Number(shape[0])
  const columnCount = Number(shape[1])
  return [
    Number.isFinite(rowCount) && rowCount >= 0 ? rowCount : 0,
    Number.isFinite(columnCount) && columnCount >= 0 ? columnCount : 0,
  ]
}

export function normalizeDataInfo(dataset) {
  if (!dataset || typeof dataset !== 'object') {
    return null
  }

  const columns = toArray(dataset.columns).map((column) => String(column))
  const dtypes = toObject(dataset.dtypes)
  const preview = toArray(dataset.preview).map((row) => {
    const safeRow = toObject(row)
    return columns.reduce((result, column) => {
      result[column] = safeRow[column]
      return result
    }, {})
  })
  const history = toArray(dataset.history).map((item, index) => {
    const record = toObject(item)
    return {
      label: toDisplayString(record.label, `处理步骤 ${index + 1}`),
      detail: toDisplayString(record.detail, '本次操作未返回详细说明。'),
      created_at: toDisplayString(record.created_at, '刚刚'),
    }
  })
  const shape = Array.isArray(dataset.shape)
    ? getSafeShape(dataset)
    : [preview.length, columns.length]

  const numericColumns = toArray(dataset.numeric_columns)
    .map((column) => String(column))
    .filter((column) => columns.includes(column))
  const inferredNumericColumns = numericColumns.length
    ? numericColumns
    : columns.filter((column) => isNumericDtype(dtypes[column]))
  const categoricalColumns = toArray(dataset.categorical_columns)
    .map((column) => String(column))
    .filter((column) => columns.includes(column))
  const inferredCategoricalColumns = categoricalColumns.length
    ? categoricalColumns
    : columns.filter((column) => !inferredNumericColumns.includes(column))

  const columnProfilesSource = toArray(dataset.column_profiles)
  const columnProfiles = (
    columnProfilesSource.length
      ? columnProfilesSource
      : columns.map((column) => ({
          column,
          dtype: String(dtypes[column] ?? ''),
          non_null: preview.length,
          missing: 0,
          unique: '-',
          sample_values: preview
            .map((row) => row?.[column])
            .filter((value) => value !== undefined && value !== null)
            .slice(0, 3),
        }))
  ).map((profile) => {
    const record = toObject(profile)
    return {
      column: toDisplayString(record.column, '-'),
      dtype: toDisplayString(record.dtype, ''),
      non_null: Number.isFinite(Number(record.non_null)) ? Number(record.non_null) : preview.length,
      missing: Number.isFinite(Number(record.missing)) ? Number(record.missing) : 0,
      unique:
        record.unique === '-' || record.unique === undefined || record.unique === null
          ? '-'
          : Number.isFinite(Number(record.unique))
            ? Number(record.unique)
            : toDisplayString(record.unique, '-'),
      sample_values: toArray(record.sample_values).map((value) => toDisplayString(value)).slice(0, 3),
    }
  })

  return {
    ...dataset,
    filename: dataset.filename ?? '当前数据集',
    shape,
    columns,
    dtypes,
    preview,
    history,
    numeric_columns: inferredNumericColumns,
    categorical_columns: inferredCategoricalColumns,
    column_profiles: columnProfiles,
    numeric_summary: toArray(dataset.numeric_summary).map((item) => ({
      ...toObject(item),
      column: toDisplayString(item?.column, '-'),
    })),
    missing_summary: toArray(dataset.missing_summary).map((item) => ({
      ...toObject(item),
      column: toDisplayString(item?.column, '-'),
    })),
  }
}

export function extractDatasetPayload(payload) {
  if (!payload || typeof payload !== 'object') {
    return null
  }
  return normalizeDataInfo(payload.dataset ?? payload)
}

export function normalizeAnalysisResult(result) {
  if (!result || typeof result !== 'object') {
    return null
  }

  return {
    ...result,
    insights: toArray(result.insights).map((item) => toDisplayString(item)),
    visualizations: toArray(result.visualizations).map((item) => toObject(item)),
    tables: toArray(result.tables).map((table) => ({
      ...toObject(table),
      title: toDisplayString(table?.title, '结果表'),
      columns: toArray(table?.columns).map((column, index) => ({
        key: toDisplayString(column?.key, `col_${index + 1}`),
        label: toDisplayString(column?.label, toDisplayString(column?.key, `列 ${index + 1}`)),
      })),
      rows: toArray(table?.rows).map((row) => toObject(row)),
    })),
    summary: toDisplayString(result.summary, '统计分析已完成。'),
    title: toDisplayString(result.title, '分析结果'),
    generated_at: toDisplayString(result.generated_at, '刚刚'),
  }
}

export function normalizeReportResult(report) {
  if (!report || typeof report !== 'object') {
    return null
  }

  return {
    ...report,
    title: toDisplayString(report.title, '分析报告'),
    generated_at: toDisplayString(report.generated_at, '刚刚'),
    html: toDisplayString(report.html, ''),
    markdown: toDisplayString(report.markdown, ''),
    sections: toArray(report.sections).map((section, index) => ({
      title: toDisplayString(section?.title, `章节 ${index + 1}`),
      body: toDisplayString(section?.body, '暂无内容。'),
      bullets: toArray(section?.bullets).map((item) => toDisplayString(item)),
    })),
  }
}

export function StatChip({ icon, label, value }) {
  return (
    <Paper
      variant="outlined"
      sx={{
        px: 1.5,
        py: 1.25,
        borderRadius: RADII.inset,
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        backgroundColor: 'rgba(255, 252, 246, 0.85)',
      }}
    >
      <Box sx={{ color: 'primary.main', display: 'flex' }}>{icon}</Box>
      <Box>
        <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
          {label}
        </Typography>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          {value}
        </Typography>
      </Box>
    </Paper>
  )
}

export function SectionHeader({ chip, title, description, icon }) {
  return (
    <Stack spacing={1.25}>
      <Chip
        icon={icon}
        label={chip}
        size="small"
        sx={{
          alignSelf: 'flex-start',
          backgroundColor: 'rgba(217, 119, 87, 0.10)',
          color: 'primary.main',
          fontWeight: 700,
        }}
      />
      <Box>
        <Typography variant="h5" sx={{ mb: 0.75 }}>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.8 }}>
          {description}
        </Typography>
      </Box>
    </Stack>
  )
}
