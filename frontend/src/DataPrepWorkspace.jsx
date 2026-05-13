import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  CircularProgress,
  FormControl,
  FormHelperText,
  InputLabel,
  ListItemText,
  MenuItem,
  OutlinedInput,
  Paper,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import DatasetIcon from '@mui/icons-material/Dataset'
import RuleFolderIcon from '@mui/icons-material/RuleFolder'
import TableChartIcon from '@mui/icons-material/TableChart'
import { RADII, SectionHeader, StatChip, TASK_LABELS, formatNumber, getSafeShape, renderSelectedFeatures } from './studioShared.jsx'

const ACTION_OPTIONS = [
  { value: 'handle_missing', label: '缺失值处理' },
  { value: 'handle_outlier', label: '异常值处理' },
  { value: 'encode_columns', label: '编码转换' },
  { value: 'scale_columns', label: '标准化' },
  { value: 'cast_columns', label: '字段类型' },
  { value: 'filter_rows', label: '筛选行' },
  { value: 'derive_column', label: '派生变量' },
  { value: 'reset_data', label: '重置数据' },
]

const isNumericDtype = (dtype) => /int|float|double|decimal|number/i.test(String(dtype ?? ''))
const isDatetimeLike = (dtype, values) =>
  /date|time/i.test(String(dtype ?? '')) ||
  (Array.isArray(values) && values.some((value) => /^\d{4}[-/]\d{1,2}[-/]\d{1,2}/.test(String(value ?? ''))))

function getMeasurementLevel({ dtype, uniqueCount, isNumeric, isDateLike }) {
  if (isDateLike) return '时间'
  if (isNumeric) return uniqueCount !== null && uniqueCount <= 12 ? '离散数值' : '连续数值'
  if (uniqueCount === 2) return '二元分类'
  if (/category/i.test(String(dtype ?? '')) || (uniqueCount !== null && uniqueCount <= 20)) return '名义分类'
  return '文本字段'
}

function getRecommendedAction({ missing, isNumeric, isDateLike, uniqueCount }) {
  if (missing > 0) return '优先补齐缺失值'
  if (isDateLike) return '建议转换为日期时间'
  if (isNumeric) return uniqueCount !== null && uniqueCount <= 12 ? '检查异常值或转等级变量' : '建议标准化并检查异常值'
  return uniqueCount !== null && uniqueCount <= 20 ? '建议编码为分类变量' : '保留文本或先筛选/派生'
}

function getQuickActions({ column, missing, isNumeric, isDateLike, uniqueCount }) {
  const actions = []
  if (missing > 0) actions.push({ label: '补缺失', preset: { action: 'handle_missing', columns: [column], strategy: isNumeric ? 'fill_median' : 'fill_mode' } })
  if (isDateLike) {
    actions.push({ label: '转日期', preset: { action: 'cast_columns', columns: [column], target_dtype: 'datetime' } })
  } else if (isNumeric) {
    actions.push({ label: '异常值', preset: { action: 'handle_outlier', columns: [column], method: 'clip_iqr', threshold: '1.5' } })
    actions.push({ label: '标准化', preset: { action: 'scale_columns', columns: [column], method: 'standardize' } })
  } else {
    actions.push({ label: '编码', preset: { action: 'encode_columns', columns: [column], method: uniqueCount !== null && uniqueCount <= 6 ? 'one_hot' : 'label' } })
    actions.push({ label: '转分类', preset: { action: 'cast_columns', columns: [column], target_dtype: 'category' } })
  }
  return actions.slice(0, 2)
}

function renderProcessFields({ dataInfo, columns, numericColumns, categoricalColumns, processForm, onProcessFormChange }) {
  switch (processForm.action) {
    case 'handle_missing':
      return (
        <>
          <FormControl fullWidth>
            <InputLabel>处理方式</InputLabel>
            <Select value={processForm.strategy} label="处理方式" onChange={(event) => onProcessFormChange('strategy', event.target.value)}>
              <MenuItem value="drop_rows">删除缺失行</MenuItem>
              <MenuItem value="fill_mean">均值填补</MenuItem>
              <MenuItem value="fill_median">中位数填补</MenuItem>
              <MenuItem value="fill_mode">众数填补</MenuItem>
              <MenuItem value="fill_value">固定值填补</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>字段</InputLabel>
            <Select multiple value={processForm.columns} label="字段" input={<OutlinedInput label="字段" />} renderValue={renderSelectedFeatures} onChange={(event) => onProcessFormChange('columns', event.target.value)}>
              {columns.map((column) => (
                <MenuItem key={column} value={column}>
                  <Checkbox checked={processForm.columns.includes(column)} />
                  <ListItemText primary={column} secondary={dataInfo?.dtypes?.[column]} />
                </MenuItem>
              ))}
            </Select>
            <FormHelperText>不选择字段时默认对全部字段执行。</FormHelperText>
          </FormControl>
          {processForm.strategy === 'fill_value' && <TextField fullWidth label="固定值" value={processForm.fill_value} onChange={(event) => onProcessFormChange('fill_value', event.target.value)} />}
        </>
      )
    case 'handle_outlier':
      return (
        <>
          <FormControl fullWidth>
            <InputLabel>异常值方式</InputLabel>
            <Select value={processForm.method} label="异常值方式" onChange={(event) => onProcessFormChange('method', event.target.value)}>
              <MenuItem value="clip_iqr">IQR 截尾</MenuItem>
              <MenuItem value="remove_iqr">IQR 删除</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>数值字段</InputLabel>
            <Select multiple value={processForm.columns} label="数值字段" input={<OutlinedInput label="数值字段" />} renderValue={renderSelectedFeatures} onChange={(event) => onProcessFormChange('columns', event.target.value)}>
              {numericColumns.map((column) => (
                <MenuItem key={column} value={column}>
                  <Checkbox checked={processForm.columns.includes(column)} />
                  <ListItemText primary={column} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField fullWidth type="number" label="IQR 阈值" value={processForm.threshold} onChange={(event) => onProcessFormChange('threshold', event.target.value)} />
        </>
      )
    case 'encode_columns':
      return (
        <>
          <FormControl fullWidth>
            <InputLabel>编码方式</InputLabel>
            <Select value={processForm.method} label="编码方式" onChange={(event) => onProcessFormChange('method', event.target.value)}>
              <MenuItem value="label">标签编码</MenuItem>
              <MenuItem value="one_hot">独热编码</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>分类字段</InputLabel>
            <Select multiple value={processForm.columns} label="分类字段" input={<OutlinedInput label="分类字段" />} renderValue={renderSelectedFeatures} onChange={(event) => onProcessFormChange('columns', event.target.value)}>
              {categoricalColumns.map((column) => (
                <MenuItem key={column} value={column}>
                  <Checkbox checked={processForm.columns.includes(column)} />
                  <ListItemText primary={column} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </>
      )
    case 'scale_columns':
      return (
        <>
          <FormControl fullWidth>
            <InputLabel>缩放方式</InputLabel>
            <Select value={processForm.method} label="缩放方式" onChange={(event) => onProcessFormChange('method', event.target.value)}>
              <MenuItem value="standardize">标准化</MenuItem>
              <MenuItem value="minmax">Min-Max</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>数值字段</InputLabel>
            <Select multiple value={processForm.columns} label="数值字段" input={<OutlinedInput label="数值字段" />} renderValue={renderSelectedFeatures} onChange={(event) => onProcessFormChange('columns', event.target.value)}>
              {numericColumns.map((column) => (
                <MenuItem key={column} value={column}>
                  <Checkbox checked={processForm.columns.includes(column)} />
                  <ListItemText primary={column} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </>
      )
    case 'cast_columns':
      return (
        <>
          <FormControl fullWidth>
            <InputLabel>目标类型</InputLabel>
            <Select value={processForm.target_dtype} label="目标类型" onChange={(event) => onProcessFormChange('target_dtype', event.target.value)}>
              <MenuItem value="number">数值</MenuItem>
              <MenuItem value="string">字符串</MenuItem>
              <MenuItem value="category">分类</MenuItem>
              <MenuItem value="datetime">日期时间</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>字段</InputLabel>
            <Select multiple value={processForm.columns} label="字段" input={<OutlinedInput label="字段" />} renderValue={renderSelectedFeatures} onChange={(event) => onProcessFormChange('columns', event.target.value)}>
              {columns.map((column) => (
                <MenuItem key={column} value={column}>
                  <Checkbox checked={processForm.columns.includes(column)} />
                  <ListItemText primary={column} secondary={dataInfo?.dtypes?.[column]} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </>
      )
    case 'filter_rows':
      return <TextField fullWidth label="筛选表达式" value={processForm.filter_expression} onChange={(event) => onProcessFormChange('filter_expression', event.target.value)} helperText="示例：age >= 25 and income > 8000" />
    case 'derive_column':
      return (
        <>
          <TextField fullWidth label="新字段名称" value={processForm.new_column} onChange={(event) => onProcessFormChange('new_column', event.target.value)} />
          <TextField fullWidth label="计算表达式" value={processForm.expression} onChange={(event) => onProcessFormChange('expression', event.target.value)} helperText="示例：(income - age) / 10" />
        </>
      )
    case 'reset_data':
      return <Alert severity="info">重置后会恢复到原始上传数据，并保留一条“重置数据”历史记录。</Alert>
    default:
      return null
  }
}

export default function DataPrepWorkspace(props) {
  const {
    file,
    loading,
    onFileChange,
    onUpload,
    dataInfo,
    taskType,
    processForm,
    onProcessFormChange,
    onApplyProcess,
    processLoading,
    processError,
    previewTableMinWidth,
    onLoadProcessPreset,
  } = props
  const history = Array.isArray(dataInfo?.history) ? dataInfo.history : []
  const profiles = Array.isArray(dataInfo?.column_profiles) ? dataInfo.column_profiles : []
  const numericSummary = Array.isArray(dataInfo?.numeric_summary) ? dataInfo.numeric_summary : []
  const missingSummary = Array.isArray(dataInfo?.missing_summary) ? dataInfo.missing_summary : []
  const columns = Array.isArray(dataInfo?.columns) ? dataInfo.columns : []
  const previewRows = Array.isArray(dataInfo?.preview) ? dataInfo.preview : []
  const shape = dataInfo ? getSafeShape(dataInfo) : [previewRows.length, columns.length]
  const numericColumns = Array.isArray(dataInfo?.numeric_columns) ? dataInfo.numeric_columns : []
  const categoricalColumns = Array.isArray(dataInfo?.categorical_columns) ? dataInfo.categorical_columns : []
  const previewRowCount = Number.isFinite(Number(dataInfo?.preview_row_count))
    ? Number(dataInfo.preview_row_count)
    : previewRows.length
  const previewTotalRows = Number.isFinite(Number(dataInfo?.preview_total_rows))
    ? Number(dataInfo.preview_total_rows)
    : shape[0]
  const previewTruncated = Boolean(dataInfo?.preview_truncated) || previewRowCount < previewTotalRows
  const totalRows = Math.max(shape[0], 1)
  const processColumns =
    processForm.action === 'handle_outlier' || processForm.action === 'scale_columns'
      ? numericColumns
      : processForm.action === 'encode_columns'
        ? categoricalColumns
        : columns
  const variableRows = profiles.map((profile) => {
    const uniqueValue = Number(profile.unique)
    const uniqueCount = Number.isFinite(uniqueValue) ? uniqueValue : null
    const isNumeric = numericColumns.includes(profile.column) || isNumericDtype(profile.dtype)
    const isDateLike = isDatetimeLike(profile.dtype, profile.sample_values)
    const missing = Number(profile.missing ?? 0)
    const missingRate = Number(((missing / totalRows) * 100).toFixed(1))
    return {
      ...profile,
      uniqueCount,
      missingRate,
      measurementLevel: getMeasurementLevel({ dtype: profile.dtype, uniqueCount, isNumeric, isDateLike }),
      recommendedAction: getRecommendedAction({ missing, isNumeric, isDateLike, uniqueCount }),
      quickActions: getQuickActions({ column: profile.column, missing, isNumeric, isDateLike, uniqueCount }),
    }
  })
  const missingFieldCount = variableRows.filter((item) => Number(item.missing ?? 0) > 0).length
  const highMissingFieldCount = variableRows.filter((item) => item.missingRate >= 20).length
  const dateCandidateCount = variableRows.filter((item) => item.measurementLevel === '时间').length
  const categoricalFieldCount = variableRows.filter(
    (item) =>
      item.measurementLevel === '名义分类' ||
      item.measurementLevel === '二元分类' ||
      item.measurementLevel === '文本字段',
  ).length
  const suggestions = [
    missingFieldCount > 0 ? `当前有 ${missingFieldCount} 个字段存在缺失值，建议先做缺失值处理再进入建模。` : null,
    highMissingFieldCount > 0 ? `有 ${highMissingFieldCount} 个字段缺失率超过 20%，建议优先评估保留价值。` : null,
    categoricalFieldCount > 0 ? `检测到 ${categoricalFieldCount} 个分类/文本字段，可在变量视图中直接预置编码。` : null,
    dateCandidateCount > 0 ? `发现 ${dateCandidateCount} 个疑似日期字段，建议先转为日期时间类型。` : null,
    numericColumns.length >= 3 ? '数值字段较多，后续可继续做标准化、相关分析、PCA 或聚类分析。' : null,
  ].filter(Boolean)
  const renderSampleValues = (values) =>
    !Array.isArray(values) || values.length === 0 ? '-' : values.map((value) => String(value)).join('，')
  const renderPreviewValue = (row, column) =>
    row?.[column] === undefined || row?.[column] === null || row?.[column] === '' ? '-' : String(row[column])

  return (
    <Stack spacing={3}>
      <Card sx={{ borderRadius: RADII.card }}>
        <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
          <Stack spacing={2.25}>
            <SectionHeader
              chip="数据准备"
              title="导入与预览数据"
              description="先把数据集导入并预览表结构，再检查字段画像和变量视图，最后再做缺失值、异常值、编码、标准化等处理。"
              icon={<CloudUploadIcon fontSize="small" />}
            />
            <Stack
              direction={{ xs: 'column', lg: 'row' }}
              spacing={2}
              alignItems={{ xs: 'stretch', lg: 'center' }}
              justifyContent="space-between"
            >
              <Typography variant="body2" color="text.secondary">
                {file ? `已选择文件：${file.name}` : '暂未选择文件'}
              </Typography>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} alignItems={{ xs: 'stretch', sm: 'center' }}>
                <input accept=".csv,.xlsx,.xls" style={{ display: 'none' }} id="dataset-upload" type="file" onChange={onFileChange} />
                <label htmlFor="dataset-upload">
                  <Button variant="contained" component="span" startIcon={<CloudUploadIcon />} size="large">
                    选择数据集
                  </Button>
                </label>
                <Button variant="outlined" color="primary" onClick={onUpload} disabled={!file || loading} size="large">
                  {loading ? <CircularProgress size={22} color="inherit" /> : '上传并预览'}
                </Button>
              </Stack>
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {dataInfo ? (
        <>
          <Card sx={{ borderRadius: RADII.card }}>
            <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
              <Stack spacing={2.25}>
                <SectionHeader
                  chip="数据预览"
                  title={previewTruncated ? '长表预览' : '全表预览'}
                  description="先看表格内容和字段分布，再决定是否处理。这里支持左右和上下滚动，并固定左侧序号列。"
                  icon={<TableChartIcon fontSize="small" />}
                />
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.25} useFlexGap flexWrap="wrap">
                  <StatChip icon={<DatasetIcon fontSize="small" />} label="文件名" value={dataInfo.filename} />
                  <StatChip icon={<DatasetIcon fontSize="small" />} label="总行数" value={`${shape[0]} 行`} />
                  <StatChip icon={<DatasetIcon fontSize="small" />} label="总列数" value={`${shape[1]} 列`} />
                  <StatChip icon={<AutoFixHighIcon fontSize="small" />} label="推荐任务" value={TASK_LABELS[taskType]} />
                </Stack>
                <Typography variant="caption" color="text.secondary">
                  {previewTruncated
                    ? `当前已加载前 ${previewRowCount} 行，共 ${previewTotalRows} 行；表格支持上下左右滚动。`
                    : `当前已加载全部 ${previewRowCount} 行；表格支持上下左右滚动。`}
                </Typography>
                <TableContainer
                  sx={{
                    maxHeight: 520,
                    overflowX: 'auto',
                    overflowY: 'auto',
                    scrollbarGutter: 'stable both-edges',
                    borderRadius: RADII.inset,
                    border: '1px solid #e8e6dc',
                    backgroundColor: '#fffdf8',
                  }}
                >
                  <Table
                    stickyHeader
                    size="small"
                    sx={{
                      minWidth: previewTableMinWidth,
                      width: 'max-content',
                      '& .sticky-index': {
                        position: 'sticky',
                        left: 0,
                        zIndex: 3,
                        backgroundColor: '#fffdf8',
                      },
                    }}
                  >
                    <TableHead>
                      <TableRow>
                        <TableCell className="sticky-index" sx={{ minWidth: 72, width: 72 }}>
                          #
                        </TableCell>
                        {columns.map((column) => (
                          <TableCell key={column} sx={{ whiteSpace: 'nowrap', minWidth: 160 }}>
                            {column}
                          </TableCell>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {previewRows.map((row, rowIndex) => (
                        <TableRow key={rowIndex} hover>
                          <TableCell className="sticky-index" sx={{ minWidth: 72, width: 72 }}>
                            {rowIndex + 1}
                          </TableCell>
                          {columns.map((column) => (
                            <TableCell key={column} sx={{ whiteSpace: 'nowrap', minWidth: 160 }}>
                              {renderPreviewValue(row, column)}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Stack>
            </CardContent>
          </Card>

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', xl: '360px minmax(0, 1fr)' },
              gap: 3,
              alignItems: 'start',
            }}
          >
            <Card sx={{ borderRadius: RADII.card }}>
              <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
                <Stack spacing={2.25}>
                  <SectionHeader
                    chip="数据质量"
                    title="数据画像与处理建议"
                    description="先识别缺失、字段类型和可疑日期列，再决定下一步处理动作。"
                    icon={<DatasetIcon fontSize="small" />}
                  />
                  <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.25} useFlexGap flexWrap="wrap">
                    <StatChip icon={<DatasetIcon fontSize="small" />} label="缺失字段" value={`${missingFieldCount} 个`} />
                    <StatChip icon={<DatasetIcon fontSize="small" />} label="高缺失字段" value={`${highMissingFieldCount} 个`} />
                    <StatChip icon={<DatasetIcon fontSize="small" />} label="分类字段" value={`${categoricalFieldCount} 个`} />
                    <StatChip icon={<DatasetIcon fontSize="small" />} label="日期候选" value={`${dateCandidateCount} 个`} />
                  </Stack>

                  {suggestions.length ? (
                    <Paper variant="outlined" sx={{ p: 1.75, borderRadius: RADII.inset }}>
                      {suggestions.map((item, index) => (
                        <Typography key={`${item}-${index}`} variant="body2" color="text.secondary" sx={{ mb: 0.8 }}>
                          {index + 1}. {item}
                        </Typography>
                      ))}
                    </Paper>
                  ) : (
                    <Alert severity="success">当前数据集没有明显的缺失或类型风险，可以直接进入建模或统计分析。</Alert>
                  )}

                  {missingSummary.length > 0 && (
                    <Paper variant="outlined" sx={{ p: 1.75, borderRadius: RADII.inset }}>
                      <Typography variant="h6" sx={{ mb: 1 }}>
                        缺失分布 Top 6
                      </Typography>
                      {missingSummary.slice(0, 6).map((item) => (
                        <Typography key={item.column} variant="body2" color="text.secondary" sx={{ mb: 0.75 }}>
                          {item.column}：缺失 {item.missing} 条，占比 {item.missing_rate}%
                        </Typography>
                      ))}
                    </Paper>
                  )}

                  {numericSummary.length > 0 && (
                    <Paper variant="outlined" sx={{ p: 1.75, borderRadius: RADII.inset }}>
                      <Typography variant="h6" sx={{ mb: 1.25 }}>
                        数值摘要
                      </Typography>
                      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.25} useFlexGap flexWrap="wrap">
                        {numericSummary.slice(0, 6).map((row) => (
                          <StatChip
                            key={row.column}
                            icon={<DatasetIcon fontSize="small" />}
                            label={row.column}
                            value={`均值 ${formatNumber(row.mean)} / 范围 ${formatNumber(row.min)}~${formatNumber(row.max)}`}
                          />
                        ))}
                      </Stack>
                    </Paper>
                  )}
                </Stack>
              </CardContent>
            </Card>

            <Card sx={{ borderRadius: RADII.card }}>
              <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
                <Stack spacing={2.25}>
                  <SectionHeader
                    chip="变量视图"
                    title="字段管理与快捷处理"
                    description="统一查看字段类型、测量层级、缺失率、样例值，并把处理动作预置到下面的处理面板。"
                    icon={<AutoFixHighIcon fontSize="small" />}
                  />
                  <Typography variant="caption" color="text.secondary">
                    行内按钮不会直接修改数据，只会把字段与处理参数加载到下方“数据处理模块”，你确认后再执行。
                  </Typography>
                  <TableContainer
                    sx={{
                      maxHeight: 520,
                      overflow: 'auto',
                      borderRadius: RADII.inset,
                      border: '1px solid #e8e6dc',
                      backgroundColor: 'rgba(255, 253, 248, 0.96)',
                    }}
                  >
                    <Table size="small" stickyHeader sx={{ minWidth: 1220 }}>
                      <TableHead>
                        <TableRow>
                          <TableCell>变量名</TableCell>
                          <TableCell>数据类型</TableCell>
                          <TableCell>测量层级</TableCell>
                          <TableCell align="right">缺失率</TableCell>
                          <TableCell align="right">唯一值</TableCell>
                          <TableCell>样例值</TableCell>
                          <TableCell>推荐处理</TableCell>
                          <TableCell sx={{ minWidth: 220 }}>快捷操作</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {variableRows.map((row) => (
                          <TableRow key={row.column} hover>
                            <TableCell sx={{ fontWeight: 600 }}>{row.column}</TableCell>
                            <TableCell>{row.dtype}</TableCell>
                            <TableCell>{row.measurementLevel}</TableCell>
                            <TableCell align="right">{row.missingRate}%</TableCell>
                            <TableCell align="right">{row.uniqueCount ?? row.unique ?? '-'}</TableCell>
                            <TableCell sx={{ minWidth: 220 }}>{renderSampleValues(row.sample_values)}</TableCell>
                            <TableCell sx={{ color: 'text.secondary', minWidth: 200 }}>{row.recommendedAction}</TableCell>
                            <TableCell>
                              {row.quickActions.length ? (
                                <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
                                  {row.quickActions.map((action) => (
                                    <Button
                                      key={`${row.column}-${action.label}`}
                                      size="small"
                                      variant="outlined"
                                      onClick={() => onLoadProcessPreset?.(action.preset)}
                                    >
                                      {action.label}
                                    </Button>
                                  ))}
                                </Stack>
                              ) : (
                                '-'
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Stack>
              </CardContent>
            </Card>
          </Box>

          <Card sx={{ borderRadius: RADII.card }}>
            <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
              <Stack spacing={2.25}>
                <SectionHeader
                  chip="处理管线"
                  title="数据处理模块"
                  description="完成上面的预览和变量检查后，再在这里正式执行缺失值、异常值、编码、标准化、筛选和派生变量处理。"
                  icon={<RuleFolderIcon fontSize="small" />}
                />

                <Box
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: { xs: '1fr', xl: 'minmax(0, 0.85fr) minmax(280px, 0.55fr)' },
                    gap: 2.5,
                    alignItems: 'start',
                  }}
                >
                  <Stack spacing={2.25}>
                    <FormControl fullWidth>
                      <InputLabel>处理类型</InputLabel>
                      <Select value={processForm.action} label="处理类型" onChange={(event) => onProcessFormChange('action', event.target.value)}>
                        {ACTION_OPTIONS.map((item) => (
                          <MenuItem key={item.value} value={item.value}>
                            {item.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    {processColumns.length > 0 && !['filter_rows', 'derive_column', 'reset_data'].includes(processForm.action) && (
                      <Alert severity="info">当前可用于此操作的字段数量：{processColumns.length}。字段越明确，处理行为越可控。</Alert>
                    )}

                    {renderProcessFields({
                      dataInfo,
                      columns,
                      numericColumns,
                      categoricalColumns,
                      processForm,
                      onProcessFormChange,
                    })}

                    {processError && <Alert severity="error">{processError}</Alert>}

                    <Button variant="contained" size="large" onClick={onApplyProcess} disabled={processLoading}>
                      {processLoading ? <CircularProgress size={22} color="inherit" /> : '应用处理'}
                    </Button>
                  </Stack>

                  <Paper variant="outlined" sx={{ p: 1.75, borderRadius: RADII.inset }}>
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      处理历史
                    </Typography>
                    {history.length ? (
                      history
                        .slice()
                        .reverse()
                        .slice(0, 6)
                        .map((item, index) => (
                          <Paper
                            key={`${item.created_at}-${index}`}
                            variant="outlined"
                            sx={{ p: 1.5, borderRadius: RADII.inset, backgroundColor: 'rgba(255, 252, 246, 0.9)', mb: 1 }}
                          >
                            <Typography variant="body2" sx={{ fontWeight: 700 }}>
                              {item.label}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {item.detail}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {item.created_at}
                            </Typography>
                          </Paper>
                        ))
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        还没有执行数据处理步骤。建议先看上面的表格预览和变量视图，再进行处理。
                      </Typography>
                    )}
                  </Paper>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </>
      ) : (
        <Alert severity="info">上传数据后，这里会先展示表格预览和变量视图，再进入数据处理流程。</Alert>
      )}
    </Stack>
  )
}
