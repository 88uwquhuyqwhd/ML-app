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
import AnalyticsIcon from '@mui/icons-material/Analytics'
import QueryStatsIcon from '@mui/icons-material/QueryStats'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { RADII, SectionHeader, formatNumber, renderSelectedFeatures } from './studioShared.jsx'

const COLORS = ['#d97757', '#6a9bcc', '#788c5d', '#d0a679']
const TOOLTIP_STYLE = {
  borderRadius: 12,
  border: '1px solid #e8e6dc',
  backgroundColor: '#fffdf8',
  boxShadow: '0 12px 30px rgba(20, 20, 19, 0.08)',
}

const ANALYSIS_OPTIONS = [
  { value: 'descriptive', label: '描述统计' },
  { value: 'correlation', label: '相关分析' },
  { value: 'ttest', label: 'T 检验' },
  { value: 'anova', label: '方差分析' },
  { value: 'chi_square', label: '卡方检验' },
  { value: 'linear_regression', label: '线性回归' },
  { value: 'logistic_regression', label: 'Logit 回归' },
  { value: 'clustering', label: '聚类分析' },
  { value: 'pca', label: 'PCA' },
  { value: 'factor_analysis', label: '因子分析' },
]

function EmptyState({ text }) {
  return (
    <Box
      sx={{
        minHeight: 220,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        px: 2,
        borderRadius: RADII.inset,
        backgroundColor: 'rgba(244, 241, 232, 0.55)',
      }}
    >
      <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.8 }}>
        {text}
      </Typography>
    </Box>
  )
}

function VisualizationCard({ title, description, children }) {
  return (
    <Paper variant="outlined" sx={{ p: 2.25, borderRadius: RADII.inset, height: '100%' }}>
      <Stack spacing={1.5} sx={{ height: '100%' }}>
        <Box>
          <Typography variant="h6" sx={{ mb: 0.5 }}>
            {title}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.75 }}>
            {description}
          </Typography>
        </Box>
        <Box sx={{ flex: 1 }}>{children}</Box>
      </Stack>
    </Paper>
  )
}

function MatrixGrid({ visual }) {
  const rows = Array.isArray(visual?.rows) ? visual.rows : []
  const columns = Array.isArray(visual?.columns) ? visual.columns : []
  const cells = Array.isArray(visual?.cells) ? visual.cells : []

  if (!rows.length || !columns.length || !cells.length) {
    return <EmptyState text="当前矩阵图缺少可展示的数据。" />
  }

  const maxValue = Math.max(...cells.map((cell) => Number(cell.value) || 0), 1)

  return (
    <Box sx={{ overflowX: 'auto' }}>
      <Box
        sx={{
          minWidth: Math.max(420, columns.length * 92 + 140),
          display: 'grid',
          gridTemplateColumns: `140px repeat(${columns.length}, minmax(84px, 1fr))`,
          gap: 1,
        }}
      >
        <Paper variant="outlined" sx={{ p: 1.25, borderRadius: 1.75, backgroundColor: 'rgba(244, 241, 232, 0.9)' }}>
          <Typography variant="caption" sx={{ fontWeight: 700, color: 'text.secondary' }}>
            行 / 列
          </Typography>
        </Paper>
        {columns.map((column) => (
          <Paper
            key={column}
            variant="outlined"
            sx={{ p: 1.25, borderRadius: 1.75, textAlign: 'center', backgroundColor: 'rgba(244, 241, 232, 0.9)' }}
          >
            <Typography variant="caption" sx={{ fontWeight: 700 }}>
              {column}
            </Typography>
          </Paper>
        ))}
        {rows.map((row) => (
          <Box key={row} sx={{ display: 'contents' }}>
            <Paper variant="outlined" sx={{ p: 1.25, borderRadius: 1.75, backgroundColor: 'rgba(244, 241, 232, 0.9)' }}>
              <Typography variant="caption" sx={{ fontWeight: 700 }}>
                {row}
              </Typography>
            </Paper>
            {columns.map((column) => {
              const cell = cells.find((item) => item.row === row && item.column === column)
              const value = Number(cell?.value) || 0
              return (
                <Paper
                  key={`${row}-${column}`}
                  variant="outlined"
                  sx={{
                    p: 1.25,
                    minHeight: 72,
                    borderRadius: 1.75,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: `rgba(217, 119, 87, ${0.12 + (value / maxValue) * 0.42})`,
                    borderColor: 'rgba(217, 119, 87, 0.18)',
                  }}
                >
                  <Typography variant="body2" sx={{ fontWeight: 700 }}>
                    {cell?.label ?? '-'}
                  </Typography>
                </Paper>
              )
            })}
          </Box>
        ))}
      </Box>
    </Box>
  )
}

function RenderVisualization({ visual }) {
  if (!visual || typeof visual !== 'object') {
    return <EmptyState text="当前图表数据不可用。" />
  }

  const data = Array.isArray(visual?.data) ? visual.data : []
  const series = Array.isArray(visual?.series) ? visual.series : []
  const points = Array.isArray(visual?.points) ? visual.points : []

  if (visual.type === 'bar') {
    if (!data.length || !series.length) {
      return <EmptyState text="当前柱状图缺少可展示的数据。" />
    }

    return (
      <VisualizationCard title={visual.title} description={visual.description}>
        <Box sx={{ height: Math.max(280, data.length * (visual.layout === 'horizontal' ? 28 : 0)) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              layout={visual.layout === 'horizontal' ? 'vertical' : 'horizontal'}
              margin={{ top: 12, right: 16, left: 12, bottom: 8 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={visual.layout !== 'horizontal'}
                horizontal={visual.layout === 'horizontal'}
                stroke="#ece7db"
              />
              {visual.layout === 'horizontal' ? (
                <>
                  <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
                  <YAxis
                    dataKey={visual.x_key}
                    type="category"
                    width={120}
                    tickLine={false}
                    axisLine={false}
                    tick={{ fill: '#6f6b63', fontSize: 12 }}
                  />
                </>
              ) : (
                <>
                  <XAxis dataKey={visual.x_key} tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
                  <YAxis tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
                </>
              )}
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value) => formatNumber(value, 4)} />
              {series.length > 1 && <Legend />}
              {series.map((item, index) => (
                <Bar key={item.key} dataKey={item.key} name={item.label} fill={item.color ?? COLORS[index % COLORS.length]} radius={[8, 8, 0, 0]}>
                  {series.length === 1 &&
                    data.map((entry, entryIndex) => (
                      <Cell key={`${item.key}-${entry[visual.x_key]}-${entryIndex}`} fill={COLORS[entryIndex % COLORS.length]} />
                    ))}
                </Bar>
              ))}
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </VisualizationCard>
    )
  }

  if (visual.type === 'matrix') {
    return (
      <VisualizationCard title={visual.title} description={visual.description}>
        <MatrixGrid visual={visual} />
      </VisualizationCard>
    )
  }

  if (visual.type === 'scatter') {
    if (!points.length) {
      return <EmptyState text="当前散点图缺少可展示的数据。" />
    }

    return (
      <VisualizationCard title={visual.title} description={visual.description}>
        <Box sx={{ height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ece7db" />
              <XAxis type="number" dataKey={visual.x_key} name={visual.x_label} tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
              <YAxis type="number" dataKey={visual.y_key} name={visual.y_label} tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value) => formatNumber(value, 4)} />
              {visual.series_key ? (
                Array.from(new Set(points.map((item) => item[visual.series_key]))).map((seriesName, index) => (
                  <Scatter
                    key={seriesName}
                    data={points.filter((item) => item[visual.series_key] === seriesName)}
                    name={seriesName}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))
              ) : (
                <Scatter data={points} fill="#d97757" />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        </Box>
      </VisualizationCard>
    )
  }

  return <EmptyState text="当前图表类型暂未支持渲染。" />
}

function LatestAnalysisResult({ result }) {
  if (!result) {
    return <EmptyState text="运行一次统计分析后，这里会显示图表、指标表和文字解读。" />
  }

  const insights = Array.isArray(result.insights) ? result.insights : []
  const visualizations = Array.isArray(result.visualizations) ? result.visualizations : []
  const tables = Array.isArray(result.tables) ? result.tables : []

  return (
    <Stack spacing={2.25}>
      <Alert severity="success">{result.summary}</Alert>

      {insights.length > 0 && (
        <Paper variant="outlined" sx={{ p: 2, borderRadius: RADII.inset }}>
          <Typography variant="h6" sx={{ mb: 1 }}>
            结果解读
          </Typography>
          <Stack spacing={0.75}>
            {insights.map((item, index) => (
              <Typography key={`${item}-${index}`} variant="body2" color="text.secondary">
                {index + 1}. {item}
              </Typography>
            ))}
          </Stack>
        </Paper>
      )}

      {visualizations.length > 0 && (
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, 1fr)' }, gap: 2 }}>
          {visualizations.map((visual, index) => (
            <RenderVisualization key={`${visual.title}-${index}`} visual={visual} />
          ))}
        </Box>
      )}

      {tables.map((table) => (
        <Paper key={table.title} variant="outlined" sx={{ p: 2, borderRadius: RADII.inset }}>
          <Typography variant="h6" sx={{ mb: 1 }}>
            {table.title}
          </Typography>
          <TableContainer sx={{ borderRadius: RADII.inset }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  {table.columns.map((column) => (
                    <TableCell key={column.key}>{column.label}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {table.rows.map((row, rowIndex) => (
                  <TableRow key={rowIndex} hover>
                    {table.columns.map((column) => (
                      <TableCell key={column.key}>{String(row[column.key] ?? '-')}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      ))}
    </Stack>
  )
}

export default function StatisticsPanel({
  dataInfo,
  analysisForm,
  onAnalysisFormChange,
  onRunAnalysis,
  analysisStatus,
  analysisError,
  analysisResults,
}) {
  const latestResult = analysisResults[analysisResults.length - 1] ?? null
  const numericColumns = dataInfo?.numeric_columns ?? []
  const allColumns = dataInfo?.columns ?? []

  const renderFields = () => {
    switch (analysisForm.analysis_type) {
      case 'descriptive':
      case 'correlation':
      case 'clustering':
      case 'pca':
      case 'factor_analysis':
        return (
          <>
            <FormControl fullWidth>
              <InputLabel>字段</InputLabel>
              <Select
                multiple
                value={analysisForm.columns}
                label="字段"
                input={<OutlinedInput label="字段" />}
                renderValue={renderSelectedFeatures}
                onChange={(event) => onAnalysisFormChange('columns', event.target.value)}
              >
                {numericColumns.map((column) => (
                  <MenuItem key={column} value={column}>
                    <Checkbox checked={analysisForm.columns.includes(column)} />
                    <ListItemText primary={column} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {analysisForm.analysis_type === 'clustering' && (
              <TextField
                fullWidth
                type="number"
                label="聚类数量"
                value={analysisForm.n_clusters}
                onChange={(event) => onAnalysisFormChange('n_clusters', event.target.value)}
              />
            )}
            {analysisForm.analysis_type === 'pca' && (
              <TextField
                fullWidth
                type="number"
                label="主成分数量"
                value={analysisForm.n_components}
                onChange={(event) => onAnalysisFormChange('n_components', event.target.value)}
              />
            )}
            {analysisForm.analysis_type === 'factor_analysis' && (
              <TextField
                fullWidth
                type="number"
                label="因子数量"
                value={analysisForm.n_factors}
                onChange={(event) => onAnalysisFormChange('n_factors', event.target.value)}
              />
            )}
          </>
        )
      case 'ttest':
      case 'anova':
        return (
          <>
            <FormControl fullWidth>
              <InputLabel>数值目标列</InputLabel>
              <Select
                value={analysisForm.target_column}
                label="数值目标列"
                onChange={(event) => onAnalysisFormChange('target_column', event.target.value)}
              >
                {numericColumns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>分组列</InputLabel>
              <Select
                value={analysisForm.group_column}
                label="分组列"
                onChange={(event) => onAnalysisFormChange('group_column', event.target.value)}
              >
                {allColumns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </>
        )
      case 'chi_square':
        return (
          <>
            <FormControl fullWidth>
              <InputLabel>行变量</InputLabel>
              <Select
                value={analysisForm.row_column}
                label="行变量"
                onChange={(event) => onAnalysisFormChange('row_column', event.target.value)}
              >
                {allColumns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>列变量</InputLabel>
              <Select
                value={analysisForm.column_column}
                label="列变量"
                onChange={(event) => onAnalysisFormChange('column_column', event.target.value)}
              >
                {allColumns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </>
        )
      case 'linear_regression':
      case 'logistic_regression':
        return (
          <>
            <FormControl fullWidth>
              <InputLabel>目标列</InputLabel>
              <Select
                value={analysisForm.target_column}
                label="目标列"
                onChange={(event) => onAnalysisFormChange('target_column', event.target.value)}
              >
                {allColumns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>特征列</InputLabel>
              <Select
                multiple
                value={analysisForm.feature_columns}
                label="特征列"
                input={<OutlinedInput label="特征列" />}
                renderValue={renderSelectedFeatures}
                onChange={(event) => onAnalysisFormChange('feature_columns', event.target.value)}
              >
                {allColumns
                  .filter((column) => column !== analysisForm.target_column)
                  .map((column) => (
                    <MenuItem key={column} value={column}>
                      <Checkbox checked={analysisForm.feature_columns.includes(column)} />
                      <ListItemText primary={column} secondary={dataInfo?.dtypes?.[column]} />
                    </MenuItem>
                  ))}
              </Select>
              <FormHelperText>这里更偏统计建模视角，和上面的训练模块可以互补使用。</FormHelperText>
            </FormControl>
          </>
        )
      default:
        return null
    }
  }

  return (
    <Card sx={{ borderRadius: RADII.card }}>
      <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
        <Stack spacing={2.25}>
          <SectionHeader
            chip="统计分析"
            title="统计分析模块"
            description="在机器学习训练之外，这里补上描述统计、相关分析、差异检验、回归、聚类、PCA 和因子分析，让项目真正更接近分析平台。"
            icon={<QueryStatsIcon fontSize="small" />}
          />

          {dataInfo ? (
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '360px minmax(0, 1fr)' }, gap: 3 }}>
              <Paper variant="outlined" sx={{ p: 2, borderRadius: RADII.inset }}>
                <Stack spacing={2}>
                  <FormControl fullWidth>
                    <InputLabel>分析类型</InputLabel>
                    <Select
                      value={analysisForm.analysis_type}
                      label="分析类型"
                      onChange={(event) => onAnalysisFormChange('analysis_type', event.target.value)}
                    >
                      {ANALYSIS_OPTIONS.map((item) => (
                        <MenuItem key={item.value} value={item.value}>
                          {item.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  {renderFields()}

                  {analysisError && <Alert severity="error">{analysisError}</Alert>}

                  <Button
                    variant="contained"
                    size="large"
                    onClick={onRunAnalysis}
                    disabled={analysisStatus === 'loading'}
                    startIcon={analysisStatus === 'loading' ? undefined : <AnalyticsIcon />}
                  >
                    {analysisStatus === 'loading' ? <CircularProgress size={22} color="inherit" /> : '运行分析'}
                  </Button>

                  <Paper variant="outlined" sx={{ p: 1.5, borderRadius: RADII.inset }}>
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      分析历史
                    </Typography>
                    {analysisResults.length ? (
                      <Stack spacing={1}>
                        {analysisResults
                          .slice()
                          .reverse()
                          .map((item, index) => (
                            <Paper
                              key={`${item.generated_at}-${index}`}
                              variant="outlined"
                              sx={{ p: 1.25, borderRadius: RADII.inset, backgroundColor: 'rgba(255, 252, 246, 0.9)' }}
                            >
                              <Typography variant="body2" sx={{ fontWeight: 700 }}>
                                {item.title}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {item.generated_at}
                              </Typography>
                            </Paper>
                          ))}
                      </Stack>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        还没有统计分析结果。
                      </Typography>
                    )}
                  </Paper>
                </Stack>
              </Paper>

              <LatestAnalysisResult result={latestResult} />
            </Box>
          ) : (
            <Alert severity="info">请先在“数据准备”页上传并整理数据，再运行统计分析。</Alert>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}
