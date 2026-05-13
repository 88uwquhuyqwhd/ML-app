import {
  Alert,
  Box,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Typography,
} from '@mui/material'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const COLORS = ['#d97757', '#6a9bcc', '#788c5d', '#d0a679']
const TOOLTIP_STYLE = {
  borderRadius: 12,
  border: '1px solid #e8e6dc',
  backgroundColor: '#fffdf8',
  boxShadow: '0 12px 30px rgba(20, 20, 19, 0.08)',
}

const SERIES_LABELS = {
  precision: '精确率',
  recall: '召回率',
  f1_score: 'F1',
}

const METRIC_LABELS = {
  accuracy: '准确率',
  precision: '精确率',
  recall: '召回率',
  f1_score: 'F1 分数',
  mse: '均方误差 (MSE)',
  rmse: '均方根误差 (RMSE)',
  mae: '平均绝对误差 (MAE)',
  r2_score: 'R2 分数',
  train_samples: '训练集样本数',
  test_samples: '测试集样本数',
  feature_count: '特征列数量',
  dropped_rows: '清理掉的行数',
}

function formatNumber(value, digits = 4) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return '-'
  return parsed.toFixed(Math.abs(parsed) >= 100 ? Math.min(digits, 2) : digits)
}

function formatPercent(value) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return '-'
  return `${parsed.toFixed(2)}%`
}

function getTaskType(result) {
  if (!result) return 'classification'
  if (result.task_type) return result.task_type
  return result.metrics?.type === 'Regression' ? 'regression' : 'classification'
}

function buildFallbackMetricBars(result) {
  if (!result?.metrics) return []
  const taskType = getTaskType(result)

  if (taskType === 'classification') {
    return [
      { name: '准确率', value: Number(((result.metrics.accuracy ?? 0) * 100).toFixed(2)) },
      { name: '精确率', value: Number(((result.metrics.precision ?? 0) * 100).toFixed(2)) },
      { name: '召回率', value: Number(((result.metrics.recall ?? 0) * 100).toFixed(2)) },
      { name: 'F1', value: Number(((result.metrics.f1_score ?? 0) * 100).toFixed(2)) },
    ]
  }

  return [
    { name: 'RMSE', value: Number((result.metrics.rmse ?? 0).toFixed(4)) },
    { name: 'MAE', value: Number((result.metrics.mae ?? 0).toFixed(4)) },
    { name: 'R2', value: Number((result.metrics.r2_score ?? 0).toFixed(4)) },
  ]
}

function getVisualizations(result) {
  const visualizations = result?.visualizations ?? {}
  return {
    metricBars: Array.isArray(visualizations.metric_bars) ? visualizations.metric_bars : buildFallbackMetricBars(result),
    featureEffects: Array.isArray(visualizations.feature_effects) ? visualizations.feature_effects : [],
    classPerformance: Array.isArray(visualizations.class_performance) ? visualizations.class_performance : [],
    confusionMatrix: Array.isArray(visualizations.confusion_matrix) ? visualizations.confusion_matrix : [],
    predictionScatter: Array.isArray(visualizations.prediction_scatter) ? visualizations.prediction_scatter : [],
  }
}

function getMetricRows(result) {
  if (!result) return []
  const taskType = getTaskType(result)
  const rows =
    taskType === 'classification'
      ? [
          ['accuracy', formatPercent((result.metrics?.accuracy ?? 0) * 100)],
          ['precision', formatPercent((result.metrics?.precision ?? 0) * 100)],
          ['recall', formatPercent((result.metrics?.recall ?? 0) * 100)],
          ['f1_score', formatPercent((result.metrics?.f1_score ?? 0) * 100)],
        ]
      : [
          ['mse', formatNumber(result.metrics?.mse, 4)],
          ['rmse', formatNumber(result.metrics?.rmse, 4)],
          ['mae', formatNumber(result.metrics?.mae, 4)],
          ['r2_score', formatNumber(result.metrics?.r2_score, 4)],
        ]

  return [
    ...rows,
    ['train_samples', result.train_samples ?? '-'],
    ['test_samples', result.test_samples ?? '-'],
    ['feature_count', result.feature_count ?? '-'],
    ['dropped_rows', result.dropped_rows ?? '-'],
  ]
}

function getOverview(result) {
  if (!result) return []
  return [
    ['模型算法', result.model_label ?? result.model_type ?? '已训练模型', '当前训练方案'],
    ['任务类型', getTaskType(result) === 'regression' ? '回归' : '分类', '基于目标列推断或手动切换'],
    ['目标列', result.target_column ?? '-', '模型最终要预测的字段'],
    ['样本划分', `${result.train_samples ?? 0} / ${result.test_samples ?? 0}`, '训练集 / 测试集'],
    ['特征数量', `${result.feature_count ?? 0} 个`, '参与训练的输入列'],
    ['清理行数', `${result.dropped_rows ?? 0} 行`, '训练前剔除的无效记录'],
  ]
}

function getHeadlineStats(result) {
  if (!result) return []

  if (getTaskType(result) === 'classification') {
    return [
      ['测试集准确率', formatPercent((result.metrics?.accuracy ?? 0) * 100), '先看整体识别正确比例'],
      ['加权 F1', formatPercent((result.metrics?.f1_score ?? 0) * 100), '兼顾精确率与召回率'],
      ['加权召回率', formatPercent((result.metrics?.recall ?? 0) * 100), '衡量漏判情况'],
    ]
  }

  return [
    ['R2', formatNumber(result.metrics?.r2_score, 4), '越接近 1 拟合越好'],
    ['RMSE', formatNumber(result.metrics?.rmse, 4), '越低越好'],
    ['MAE', formatNumber(result.metrics?.mae, 4), '平均误差水平'],
  ]
}

function getBounds(points) {
  const values = points
    .flatMap((point) => [point.actual, point.predicted])
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
  if (!values.length) return null
  return { min: Math.min(...values), max: Math.max(...values) }
}

function matrixBg(count, maxCount) {
  return `rgba(217, 119, 87, ${0.12 + (count / Math.max(maxCount, 1)) * 0.42})`
}

function EmptyChart({ message, radius }) {
  return (
    <Box
      sx={{
        minHeight: 260,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        px: 2,
        py: 3,
        borderRadius: radius,
        backgroundColor: 'rgba(244, 241, 232, 0.55)',
      }}
    >
      <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 320, lineHeight: 1.8 }}>
        {message}
      </Typography>
    </Box>
  )
}

function ChartCard({ title, description, radius, children }) {
  return (
    <Paper
      variant="outlined"
      sx={{ p: 2.25, borderRadius: radius, backgroundColor: 'rgba(255, 253, 248, 0.95)', height: '100%' }}
    >
      <Stack spacing={1.75} sx={{ height: '100%' }}>
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

function ConfusionMatrix({ data, radius }) {
  if (!data.length) {
    return <EmptyChart message="当前结果暂未生成混淆矩阵，可重新训练分类模型或更换数据集后查看。" radius={radius} />
  }

  const predictedLabels = Array.isArray(data[0]?.values)
    ? data[0].values.map((item) => item.predicted)
    : []
  const maxCount = Math.max(
    ...data.flatMap((row) => (Array.isArray(row.values) ? row.values.map((cell) => cell.count) : [])),
    1,
  )

  return (
    <Stack spacing={1.25}>
      <Typography variant="caption" color="text.secondary">
        纵轴表示实际类别，横轴表示预测类别。
      </Typography>
      <Box sx={{ overflowX: 'auto' }}>
        <Box
          sx={{
            minWidth: Math.max(420, predictedLabels.length * 94 + 140),
            display: 'grid',
            gridTemplateColumns: `140px repeat(${predictedLabels.length}, minmax(84px, 1fr))`,
            gap: 1,
          }}
        >
          <Paper variant="outlined" sx={{ p: 1.25, borderRadius: 1.75, backgroundColor: 'rgba(244, 241, 232, 0.9)' }}>
            <Typography variant="caption" sx={{ fontWeight: 700, color: 'text.secondary' }}>
              实际 / 预测
            </Typography>
          </Paper>
          {predictedLabels.map((label) => (
            <Paper
              key={label}
              variant="outlined"
              sx={{ p: 1.25, borderRadius: 1.75, textAlign: 'center', backgroundColor: 'rgba(244, 241, 232, 0.9)' }}
            >
              <Typography variant="caption" sx={{ fontWeight: 700 }}>
                {label}
              </Typography>
            </Paper>
          ))}
          {data.map((row) => (
            <Box key={row.actual} sx={{ display: 'contents' }}>
              <Paper variant="outlined" sx={{ p: 1.25, borderRadius: 1.75, backgroundColor: 'rgba(244, 241, 232, 0.9)' }}>
                <Typography variant="caption" sx={{ fontWeight: 700 }}>
                  {row.actual}
                </Typography>
              </Paper>
              {(Array.isArray(row.values) ? row.values : []).map((cell) => (
                <Paper
                  key={`${row.actual}-${cell.predicted}`}
                  variant="outlined"
                  sx={{
                    p: 1.25,
                    minHeight: 72,
                    borderRadius: 1.75,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: matrixBg(cell.count, maxCount),
                    borderColor: 'rgba(217, 119, 87, 0.18)',
                  }}
                >
                  <Typography variant="h6" sx={{ lineHeight: 1.1 }}>
                    {cell.count}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {cell.predicted}
                  </Typography>
                </Paper>
              ))}
            </Box>
          ))}
        </Box>
      </Box>
    </Stack>
  )
}

export default function TrainingResultPanel({ trainingStatus, trainingResult, trainError, radius = '14px' }) {
  const taskType = getTaskType(trainingResult)
  const visualizations = getVisualizations(trainingResult)
  const metricRows = getMetricRows(trainingResult)
  const overview = getOverview(trainingResult)
  const headlineStats = getHeadlineStats(trainingResult)
  const bounds = getBounds(visualizations.predictionScatter)
  const classes = Array.isArray(trainingResult?.classes) ? trainingResult.classes : []

  if (trainingStatus === 'idle') {
    return <Alert severity="info">上传数据后，确认字段和任务类型，再点击“开始训练”。训练完成后，这里会自动生成图表分析。</Alert>
  }

  if (trainingStatus === 'training') {
    return <Alert severity="warning">正在训练模型，请稍候...</Alert>
  }

  if (trainingStatus === 'error') {
    return <Alert severity="error">{trainError}</Alert>
  }

  if (!trainingResult) {
    return <Alert severity="info">训练结果将在这里展示。</Alert>
  }

  return (
    <Stack spacing={2.5}>
      <Alert severity="success">{trainingResult.message}</Alert>

      <Paper
        variant="outlined"
        sx={{
          p: 2.25,
          borderRadius: radius,
          background:
            'linear-gradient(135deg, rgba(217, 119, 87, 0.10) 0%, rgba(255, 253, 248, 0.98) 54%, rgba(106, 155, 204, 0.10) 100%)',
        }}
      >
        <Stack spacing={2}>
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.75 }}>
              结果总览
            </Typography>
            <Typography variant="h5" sx={{ mb: 0.5 }}>
              {trainingResult.model_label ?? trainingResult.model_type ?? '已训练模型'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              目标列：{trainingResult.target_column ?? '-'} | 任务类型：{taskType === 'regression' ? '回归' : '分类'} | 特征列：
              {trainingResult.feature_count ?? 0} 个
            </Typography>
          </Box>

          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(3, 1fr)' }, gap: 1.25 }}>
            {headlineStats.map(([label, value, caption]) => (
              <Paper
                key={label}
                variant="outlined"
                sx={{ p: 1.75, borderRadius: radius, backgroundColor: 'rgba(255, 253, 248, 0.88)' }}
              >
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.75 }}>
                  {label}
                </Typography>
                <Typography variant="h5" sx={{ mb: 0.25 }}>
                  {value}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {caption}
                </Typography>
              </Paper>
            ))}
          </Box>
        </Stack>
      </Paper>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }, gap: 1.5 }}>
        {overview.map(([label, value, caption]) => (
          <Paper
            key={label}
            variant="outlined"
            sx={{
              p: 2,
              borderRadius: radius,
              background: 'linear-gradient(180deg, rgba(255, 254, 250, 0.98) 0%, rgba(249, 245, 236, 0.92) 100%)',
            }}
          >
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.75 }}>
              {label}
            </Typography>
            <Typography variant="h6" sx={{ mb: 0.5, wordBreak: 'break-word' }}>
              {value}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {caption}
            </Typography>
          </Paper>
        ))}
      </Box>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, 1fr)' }, gap: 2 }}>
        <ChartCard title="核心指标" description="优先看整体表现，分类任务展示百分比指标，回归任务展示误差与拟合度。" radius={radius}>
          {visualizations.metricBars.length > 0 ? (
            <Box sx={{ height: 290 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={visualizations.metricBars} margin={{ top: 12, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#ece7db" />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
                  <YAxis
                    tickLine={false}
                    axisLine={false}
                    tick={{ fill: '#6f6b63', fontSize: 12 }}
                    domain={taskType === 'classification' ? [0, 100] : ['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={TOOLTIP_STYLE}
                    formatter={(value) => [taskType === 'classification' ? formatPercent(value) : formatNumber(value, 4), '指标值']}
                  />
                  <Bar dataKey="value" radius={[10, 10, 0, 0]}>
                    {visualizations.metricBars.map((entry, index) => (
                      <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Box>
          ) : (
            <EmptyChart message="当前结果没有可展示的指标数据。" radius={radius} />
          )}
        </ChartCard>

        {taskType === 'classification' ? (
          <ChartCard title="类别表现" description="查看每个类别的精确率、召回率和 F1，能更快发现偏向某些类别的问题。" radius={radius}>
            {visualizations.classPerformance.length > 0 ? (
              <Box sx={{ height: 290 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={visualizations.classPerformance} margin={{ top: 12, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#ece7db" />
                    <XAxis dataKey="class_name" tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
                    <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      labelFormatter={(label) => `类别：${label}`}
                      formatter={(value, name, item) => [
                        formatPercent(value),
                        `${SERIES_LABELS[name] ?? name}${item?.payload?.support ? ` | 样本数 ${item.payload.support}` : ''}`,
                      ]}
                    />
                    <Legend formatter={(value) => SERIES_LABELS[value] ?? value} />
                    <Bar dataKey="precision" name="precision" fill="#d97757" radius={[8, 8, 0, 0]} />
                    <Bar dataKey="recall" name="recall" fill="#6a9bcc" radius={[8, 8, 0, 0]} />
                    <Bar dataKey="f1_score" name="f1_score" fill="#788c5d" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            ) : (
              <EmptyChart message="当前结果暂未生成分类别表现图，可重新训练后再查看类别差异。" radius={radius} />
            )}
          </ChartCard>
        ) : (
          <ChartCard title="预测散点" description="点越贴近对角线，说明预测值越接近真实值。这个图对判断回归拟合质量最直观。" radius={radius}>
            {visualizations.predictionScatter.length > 0 ? (
              <Box sx={{ height: 290 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ece7db" />
                    <XAxis type="number" dataKey="actual" name="实际值" tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
                    <YAxis type="number" dataKey="predicted" name="预测值" tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
                    <Tooltip
                      cursor={{ strokeDasharray: '4 4' }}
                      contentStyle={TOOLTIP_STYLE}
                      formatter={(value, name) => [formatNumber(value, 4), name === 'actual' ? '实际值' : '预测值']}
                    />
                    {bounds && (
                      <ReferenceLine
                        segment={[
                          { x: bounds.min, y: bounds.min },
                          { x: bounds.max, y: bounds.max },
                        ]}
                        stroke="#788c5d"
                        strokeDasharray="5 5"
                      />
                    )}
                    <Scatter data={visualizations.predictionScatter} fill="#d97757" />
                  </ScatterChart>
                </ResponsiveContainer>
              </Box>
            ) : (
              <EmptyChart message="当前结果暂未生成预测散点图，可重新训练回归模型后查看拟合分布。" radius={radius} />
            )}
          </ChartCard>
        )}

        <ChartCard title="特征影响" description="优先展示影响最大的字段，帮助你快速理解模型主要依赖哪些输入特征。" radius={radius}>
          {visualizations.featureEffects.length > 0 ? (
            <Box sx={{ height: Math.max(300, visualizations.featureEffects.length * 28) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={visualizations.featureEffects} layout="vertical" margin={{ top: 8, right: 18, left: 30, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#ece7db" />
                  <XAxis type="number" tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
                  <YAxis dataKey="feature" type="category" width={130} tickLine={false} axisLine={false} tick={{ fill: '#6f6b63', fontSize: 12 }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value) => [formatNumber(value, 6), '影响值']} />
                  <Bar dataKey="value" fill="#6a9bcc" radius={[0, 10, 10, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          ) : (
            <EmptyChart message="当前模型没有可解释的特征权重，例如某些核方法模型可能不会直接返回该信息。" radius={radius} />
          )}
        </ChartCard>

        {taskType === 'classification' && (
          <ChartCard title="混淆矩阵" description="如果某两个类别经常互相混淆，这里会非常明显，适合快速定位分类错误集中在哪些类别组合。" radius={radius}>
            <ConfusionMatrix data={visualizations.confusionMatrix} radius={radius} />
          </ChartCard>
        )}
      </Box>

      <Paper variant="outlined" sx={{ p: 2, borderRadius: radius, backgroundColor: 'rgba(255, 253, 248, 0.95)' }}>
        <Stack spacing={1.5}>
          <Box>
            <Typography variant="h6" sx={{ mb: 0.5 }}>
              结果明细
            </Typography>
            <Typography variant="body2" color="text.secondary">
              这里保留精确数值，便于和图表一起交叉判断。所有指标均基于当前测试集结果生成。
            </Typography>
          </Box>
          <TableContainer sx={{ borderRadius: radius }}>
            <Table size="small">
              <TableBody>
                {metricRows.map(([key, value]) => (
                  <TableRow key={key}>
                    <TableCell>{METRIC_LABELS[key] ?? key}</TableCell>
                    <TableCell align="right">{value}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          {classes.length > 0 && (
            <Typography variant="body2" color="text.secondary">
              类别标签：{classes.join('，')}
            </Typography>
          )}
        </Stack>
      </Paper>
    </Stack>
  )
}
