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
  Select,
  Stack,
  TextField,
} from '@mui/material'
import AutoGraphIcon from '@mui/icons-material/AutoGraph'
import DatasetIcon from '@mui/icons-material/Dataset'
import PsychologyIcon from '@mui/icons-material/Psychology'
import TuneIcon from '@mui/icons-material/Tune'
import TrainingResultPanel from './TrainingResultPanel.jsx'
import StatisticsPanel from './StatisticsPanel.jsx'
import {
  RADII,
  SectionHeader,
  StatChip,
  TASK_LABELS,
  getModelLabel,
  renderSelectedFeatures,
} from './studioShared.jsx'

export default function ModelingWorkspace({
  dataInfo,
  targetCol,
  selectedFeatures,
  taskType,
  modelType,
  hyperparameters,
  availableFeatureColumns,
  onTargetChange,
  onFeatureChange,
  onTaskTypeChange,
  onModelTypeChange,
  onHyperparameterChange,
  onTrain,
  isTrainDisabled,
  trainingStatus,
  trainingResult,
  trainError,
  analysisProps,
}) {
  const targetProfile = Array.isArray(dataInfo?.column_profiles)
    ? dataInfo.column_profiles.find((item) => item.column === targetCol)
    : null
  const targetUniqueCount = Number(targetProfile?.unique)
  const targetDtype = String(dataInfo?.dtypes?.[targetCol] ?? '').toLowerCase()
  const isNumericTarget =
    targetDtype.includes('int') ||
    targetDtype.includes('float') ||
    targetDtype.includes('double') ||
    targetDtype.includes('decimal')
  const shouldWarnContinuousClassification =
    taskType === 'classification' &&
    isNumericTarget &&
    Number.isFinite(targetUniqueCount) &&
    targetUniqueCount > 12

  const renderHyperparameterFields = () => {
    if (modelType === 'RandomForest') {
      return (
        <Stack spacing={2}>
          <TextField
            label="树的数量"
            type="number"
            value={hyperparameters.n_estimators ?? ''}
            onChange={(event) => onHyperparameterChange('n_estimators', event.target.value)}
            inputProps={{ min: 1 }}
            fullWidth
          />
          <TextField
            label="最大深度（可选）"
            type="number"
            value={hyperparameters.max_depth ?? ''}
            onChange={(event) => onHyperparameterChange('max_depth', event.target.value)}
            inputProps={{ min: 1 }}
            fullWidth
          />
        </Stack>
      )
    }

    if (modelType === 'SVM') {
      return (
        <Stack spacing={2}>
          <TextField
            label="惩罚系数 C"
            type="number"
            value={hyperparameters.C ?? ''}
            onChange={(event) => onHyperparameterChange('C', event.target.value)}
            inputProps={{ min: 0.0001, step: 0.1 }}
            fullWidth
          />
          <FormControl fullWidth>
            <InputLabel>核函数</InputLabel>
            <Select
              value={hyperparameters.kernel ?? 'rbf'}
              onChange={(event) => onHyperparameterChange('kernel', event.target.value)}
              label="核函数"
            >
              <MenuItem value="linear">线性核</MenuItem>
              <MenuItem value="rbf">RBF</MenuItem>
              <MenuItem value="poly">多项式核</MenuItem>
              <MenuItem value="sigmoid">Sigmoid</MenuItem>
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>Gamma</InputLabel>
            <Select
              value={hyperparameters.gamma ?? 'scale'}
              onChange={(event) => onHyperparameterChange('gamma', event.target.value)}
              label="Gamma"
            >
              <MenuItem value="scale">scale</MenuItem>
              <MenuItem value="auto">auto</MenuItem>
            </Select>
          </FormControl>
        </Stack>
      )
    }

    if (taskType === 'classification') {
      return (
        <Stack spacing={2}>
          <TextField
            label="惩罚系数 C"
            type="number"
            value={hyperparameters.C ?? ''}
            onChange={(event) => onHyperparameterChange('C', event.target.value)}
            inputProps={{ min: 0.0001, step: 0.1 }}
            fullWidth
          />
          <TextField
            label="最大迭代次数"
            type="number"
            value={hyperparameters.max_iter ?? ''}
            onChange={(event) => onHyperparameterChange('max_iter', event.target.value)}
            inputProps={{ min: 1 }}
            fullWidth
          />
        </Stack>
      )
    }

    return <Alert severity="info">线性回归在当前版本中不需要额外超参数。</Alert>
  }

  return (
    <Stack spacing={3}>
      {dataInfo ? (
        <>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', lg: 'minmax(340px, 400px) minmax(0, 1fr)' },
              gap: 3,
              alignItems: 'start',
            }}
          >
            <Card sx={{ borderRadius: RADII.card }}>
              <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
                <Stack spacing={2.25}>
                  <SectionHeader
                    chip="训练配置"
                    title="机器学习建模"
                    description="这里保留面向机器学习的训练工作流：选择目标列、特征列、任务类型和模型算法，然后观察结果图表。"
                    icon={<TuneIcon fontSize="small" />}
                  />

                  <Box
                    sx={{
                      p: 1.75,
                      border: '1px solid #e8e6dc',
                      borderRadius: RADII.inset,
                      backgroundColor: 'rgba(255, 252, 246, 0.88)',
                    }}
                  >
                    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.25} useFlexGap flexWrap="wrap">
                      <StatChip icon={<DatasetIcon fontSize="small" />} label="目标列" value={targetCol || '-'} />
                      <StatChip icon={<DatasetIcon fontSize="small" />} label="特征列" value={`${selectedFeatures.length} 个`} />
                      <StatChip icon={<PsychologyIcon fontSize="small" />} label="任务类型" value={TASK_LABELS[taskType]} />
                    </Stack>
                  </Box>

                  <FormControl fullWidth>
                    <InputLabel>目标列 (y)</InputLabel>
                    <Select value={targetCol} onChange={onTargetChange} label="目标列 (y)">
                      {(dataInfo?.columns ?? []).map((column) => (
                        <MenuItem key={column} value={column}>
                          {column} ({dataInfo.dtypes[column]})
                        </MenuItem>
                      ))}
                    </Select>
                    <FormHelperText>模型最终要预测的字段。</FormHelperText>
                  </FormControl>

                  <FormControl fullWidth>
                    <InputLabel>特征列 (X)</InputLabel>
                    <Select
                      multiple
                      value={selectedFeatures}
                      onChange={onFeatureChange}
                      input={<OutlinedInput label="特征列 (X)" />}
                      renderValue={renderSelectedFeatures}
                    >
                      {availableFeatureColumns.map((column) => (
                        <MenuItem key={column} value={column}>
                          <Checkbox checked={selectedFeatures.includes(column)} />
                          <ListItemText primary={column} secondary={dataInfo.dtypes[column]} />
                        </MenuItem>
                      ))}
                    </Select>
                    <FormHelperText>已选 {selectedFeatures.length} 个特征列。</FormHelperText>
                  </FormControl>

                  <FormControl fullWidth>
                    <InputLabel>任务类型</InputLabel>
                    <Select value={taskType} onChange={onTaskTypeChange} label="任务类型">
                      <MenuItem value="classification">分类</MenuItem>
                      <MenuItem value="regression">回归</MenuItem>
                    </Select>
                  </FormControl>

                  {shouldWarnContinuousClassification && (
                    <Alert severity="warning">
                      当前目标列是数值型且唯一值较多，更像连续变量。建议切换到“回归”；如果你确实想做分类，请先在数据准备里把目标列分箱成有限类别。
                    </Alert>
                  )}

                  <FormControl fullWidth>
                    <InputLabel>模型算法</InputLabel>
                    <Select value={modelType} onChange={onModelTypeChange} label="模型算法">
                      <MenuItem value="RandomForest">随机森林</MenuItem>
                      <MenuItem value="SVM">支持向量机</MenuItem>
                      <MenuItem value="LogisticRegression">{getModelLabel('LogisticRegression', taskType)}</MenuItem>
                    </Select>
                  </FormControl>

                  {renderHyperparameterFields()}

                  <Button variant="contained" size="large" onClick={onTrain} disabled={isTrainDisabled} sx={{ py: 1.4 }}>
                    {trainingStatus === 'training' ? <CircularProgress size={22} color="inherit" /> : '开始训练'}
                  </Button>
                </Stack>
              </CardContent>
            </Card>

            <Card sx={{ borderRadius: RADII.card }}>
              <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
                <Stack spacing={2.25}>
                  <SectionHeader
                    chip="训练结果"
                    title="模型结果与可视化"
                    description="结果区会展示训练摘要、核心指标、类别表现、混淆矩阵、预测散点和特征影响。"
                    icon={<AutoGraphIcon fontSize="small" />}
                  />
                  <TrainingResultPanel
                    trainingStatus={trainingStatus}
                    trainingResult={trainingResult}
                    trainError={trainError}
                    radius={RADII.inset}
                  />
                </Stack>
              </CardContent>
            </Card>
          </Box>

          <StatisticsPanel {...analysisProps} />
        </>
      ) : (
        <Alert severity="info">请先在“数据准备”页上传数据，处理完成后再进入分析建模。</Alert>
      )}
    </Stack>
  )
}
