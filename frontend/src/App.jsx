import { useEffect, useMemo, useState } from 'react'
import { Alert, Box, Button, Chip, Container, Paper, Stack, Typography } from '@mui/material'
import AutoGraphIcon from '@mui/icons-material/AutoGraph'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import FolderCopyIcon from '@mui/icons-material/FolderCopy'
import InsightsIcon from '@mui/icons-material/Insights'
import PsychologyIcon from '@mui/icons-material/Psychology'
import axios from 'axios'
import DataPrepWorkspace from './DataPrepWorkspace.jsx'
import ModelingWorkspace from './ModelingWorkspace.jsx'
import ProjectWorkspace from './ProjectWorkspace.jsx'
import ReportWorkspace from './ReportWorkspace.jsx'
import {
  API_BASE,
  DEFAULT_MODEL_TYPE,
  RADII,
  StatChip,
  extractDatasetPayload,
  formatErrorMessage,
  getAvailableFeatureColumns,
  getDefaultHyperparameters,
  inferTaskType,
  normalizeAnalysisResult,
  normalizeReportResult,
} from './studioShared.jsx'

const VIEW_ITEMS = [
  { value: 'data_prepare', label: '数据准备', icon: <CloudUploadIcon fontSize="small" /> },
  { value: 'modeling', label: '分析建模', icon: <PsychologyIcon fontSize="small" /> },
  { value: 'report', label: '结果报告', icon: <AutoGraphIcon fontSize="small" /> },
  { value: 'project', label: '项目管理', icon: <FolderCopyIcon fontSize="small" /> },
]

const BACKEND_STARTING_MESSAGE = '本地分析服务启动中，请稍候几秒后再操作。'
const BACKEND_READY_MESSAGE = '本地分析服务已连接，可以开始上传、处理、训练与生成报告。'
const BACKEND_UNAVAILABLE_MESSAGE =
  '本地分析服务尚未启动完成或已经退出。请稍候再试；如果仍然失败，请完全关闭应用后重新打开。'
const BACKEND_BOOT_TIMEOUT_MESSAGE =
  '本地分析服务启动超时。请完全关闭应用后重新打开；如果你运行的是便携版，请先完整解压压缩包再启动。'

function isNetworkError(error) {
  return error?.code === 'ERR_NETWORK' || /network error/i.test(String(error?.message ?? ''))
}

function getDefaultProcessForm(action = 'handle_missing') {
  return {
    action,
    columns: [],
    strategy: 'drop_rows',
    fill_value: '',
    method:
      action === 'handle_outlier'
        ? 'clip_iqr'
        : action === 'encode_columns'
          ? 'label'
          : 'standardize',
    threshold: '1.5',
    target_dtype: 'string',
    filter_expression: '',
    new_column: '',
    expression: '',
  }
}

function getDefaultAnalysisForm(type = 'descriptive') {
  return {
    analysis_type: type,
    columns: [],
    target_column: '',
    group_column: '',
    row_column: '',
    column_column: '',
    feature_columns: [],
    n_clusters: '3',
    n_components: '2',
    n_factors: '2',
  }
}

function loadProjectMeta() {
  try {
    const raw = window.localStorage.getItem('ml_studio_meta')
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function wait(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

async function requestBackendMeta() {
  const response = await axios.get(`${API_BASE}/api/meta`, { timeout: 3000 })
  return response.data
}

async function requestRecentProjects(limit = 8) {
  const response = await axios.get(`${API_BASE}/api/projects`, {
    params: { limit },
  })
  return Array.isArray(response.data?.items) ? response.data.items : []
}

function App() {
  const initialMeta = loadProjectMeta()
  const [activeView, setActiveView] = useState(initialMeta.activeView || 'data_prepare')
  const [projectName, setProjectName] = useState(initialMeta.projectName || '我的分析项目')
  const [projectNotes, setProjectNotes] = useState(initialMeta.projectNotes || '')

  const [file, setFile] = useState(null)
  const [dataInfo, setDataInfo] = useState(null)
  const [loading, setLoading] = useState(false)
  const [recentProjects, setRecentProjects] = useState([])
  const [backendStatus, setBackendStatus] = useState('starting')
  const [backendStatusMessage, setBackendStatusMessage] = useState(BACKEND_STARTING_MESSAGE)

  const [targetCol, setTargetCol] = useState('')
  const [selectedFeatures, setSelectedFeatures] = useState([])
  const [taskType, setTaskType] = useState('classification')
  const [modelType, setModelType] = useState(DEFAULT_MODEL_TYPE)
  const [hyperparameters, setHyperparameters] = useState(
    getDefaultHyperparameters('classification', DEFAULT_MODEL_TYPE),
  )

  const [trainingResult, setTrainingResult] = useState(null)
  const [trainingStatus, setTrainingStatus] = useState('idle')
  const [trainError, setTrainError] = useState(null)

  const [processForm, setProcessForm] = useState(getDefaultProcessForm())
  const [processLoading, setProcessLoading] = useState(false)
  const [processError, setProcessError] = useState(null)

  const [analysisForm, setAnalysisForm] = useState(getDefaultAnalysisForm())
  const [analysisStatus, setAnalysisStatus] = useState('idle')
  const [analysisError, setAnalysisError] = useState(null)
  const [analysisResults, setAnalysisResults] = useState([])

  const [reportStatus, setReportStatus] = useState('idle')
  const [reportError, setReportError] = useState(null)
  const [reportResult, setReportResult] = useState(null)

  const [projectSaveStatus, setProjectSaveStatus] = useState('idle')
  const [projectSaveError, setProjectSaveError] = useState(null)
  const [savedProjectInfo, setSavedProjectInfo] = useState(null)
  const [projectLoadStatus, setProjectLoadStatus] = useState('idle')
  const [projectLoadError, setProjectLoadError] = useState(null)
  const [loadingProjectId, setLoadingProjectId] = useState(null)

  const availableFeatureColumns = useMemo(
    () => getAvailableFeatureColumns(dataInfo, targetCol),
    [dataInfo, targetCol],
  )
  const previewTableMinWidth = useMemo(
    () => Math.max(1080, (dataInfo?.columns?.length ?? 0) * 168),
    [dataInfo],
  )

  const isBackendReady = backendStatus === 'ready'

  useEffect(() => {
    window.localStorage.setItem(
      'ml_studio_meta',
      JSON.stringify({ activeView, projectName, projectNotes }),
    )
  }, [activeView, projectName, projectNotes])

  const probeBackend = async () => {
    const meta = await requestBackendMeta()
    setBackendStatus('ready')
    setBackendStatusMessage(BACKEND_READY_MESSAGE)
    return meta
  }

  const fetchRecentProjects = async (limit = 8) => {
    const items = await requestRecentProjects(limit)
    setRecentProjects(items)
    return items
  }

  const refreshPlatformState = async () => {
    try {
      await probeBackend()
      await fetchRecentProjects(8)
    } catch {
      setRecentProjects([])
    }
  }

  const handleBackendUnavailable = (actionLabel) => {
    const message = `${actionLabel}失败：${BACKEND_UNAVAILABLE_MESSAGE}`
    setBackendStatus('error')
    setBackendStatusMessage(BACKEND_UNAVAILABLE_MESSAGE)
    return message
  }

  const ensureBackendReady = async (actionLabel) => {
    if (backendStatus === 'ready') {
      return true
    }

    setBackendStatus('starting')
    setBackendStatusMessage(BACKEND_STARTING_MESSAGE)

    try {
      await probeBackend()
      return true
    } catch {
      window.alert(handleBackendUnavailable(actionLabel))
      return false
    }
  }

  const handleRetryBackend = async () => {
    setBackendStatus('starting')
    setBackendStatusMessage(BACKEND_STARTING_MESSAGE)

    try {
      await probeBackend()
      await fetchRecentProjects(8)
    } catch {
      setBackendStatus('error')
      setBackendStatusMessage(BACKEND_UNAVAILABLE_MESSAGE)
      setRecentProjects([])
    }
  }

  useEffect(() => {
    let active = true

    const bootstrapPlatform = async () => {
      setBackendStatus('starting')
      setBackendStatusMessage(BACKEND_STARTING_MESSAGE)

      for (let attempt = 0; attempt < 18; attempt += 1) {
        try {
          if (!active) return
          await requestBackendMeta()
          if (!active) return
          const items = await requestRecentProjects(8)
          if (!active) return

          setBackendStatus('ready')
          setBackendStatusMessage(BACKEND_READY_MESSAGE)
          setRecentProjects(items)
          return
        } catch {
          if (attempt < 17) {
            await wait(900)
          }
        }
      }

      if (active) {
        setBackendStatus('error')
        setBackendStatusMessage(BACKEND_BOOT_TIMEOUT_MESSAGE)
        setRecentProjects([])
      }
    }

    bootstrapPlatform()
    return () => {
      active = false
    }
  }, [])

  const clearTrainingState = () => {
    setTrainingResult(null)
    setTrainingStatus('idle')
    setTrainError(null)
  }

  const clearAnalysisState = () => {
    setAnalysisStatus('idle')
    setAnalysisError(null)
    setAnalysisResults([])
  }

  const clearReportState = () => {
    setReportStatus('idle')
    setReportError(null)
    setReportResult(null)
  }

  const clearDerivedOutputs = () => {
    clearTrainingState()
    clearAnalysisState()
    clearReportState()
  }

  const normalizeTrainingResult = (result) => {
    if (!result || typeof result !== 'object' || Object.keys(result).length === 0) {
      return null
    }
    return result
  }

  const syncDatasetToModeling = (dataset, restoredState = null) => {
    const datasetColumns = Array.isArray(dataset?.columns) ? dataset.columns : []
    const preferredTarget =
      datasetColumns.includes(restoredState?.targetCol ?? '')
        ? restoredState.targetCol
        : datasetColumns.includes(targetCol)
          ? targetCol
          : datasetColumns[datasetColumns.length - 1] || ''
    const nextTaskType =
      restoredState?.taskType === 'classification' || restoredState?.taskType === 'regression'
        ? restoredState.taskType
        : preferredTarget
          ? inferTaskType(dataset, preferredTarget)
          : 'classification'
    const nextModelType =
      typeof restoredState?.modelType === 'string' && restoredState.modelType
        ? restoredState.modelType
        : DEFAULT_MODEL_TYPE
    const nextAvailableFeatures = getAvailableFeatureColumns(dataset, preferredTarget)
    const preferredFeatureColumns = Array.isArray(restoredState?.selectedFeatures)
      ? restoredState.selectedFeatures
      : selectedFeatures
    const nextFeatures = preferredFeatureColumns.filter(
      (feature) => datasetColumns.includes(feature) && feature !== preferredTarget,
    )
    const nextAction =
      typeof restoredState?.processForm?.action === 'string'
        ? restoredState.processForm.action
        : restoredState
          ? 'handle_missing'
          : processForm.action
    const rawAnalysisForm =
      restoredState?.analysisForm && typeof restoredState.analysisForm === 'object'
        ? restoredState.analysisForm
        : null
    const nextAnalysisType =
      typeof rawAnalysisForm?.analysis_type === 'string'
        ? rawAnalysisForm.analysis_type
        : restoredState
          ? 'descriptive'
          : analysisForm.analysis_type
    const shouldResetModel =
      !restoredState && (nextTaskType !== taskType || !datasetColumns.includes(targetCol))

    setTargetCol(preferredTarget)
    setSelectedFeatures(nextFeatures.length > 0 ? nextFeatures : nextAvailableFeatures)
    setTaskType(nextTaskType)
    setModelType(shouldResetModel ? DEFAULT_MODEL_TYPE : nextModelType)
    setHyperparameters(
      restoredState
        ? restoredState.hyperparameters && typeof restoredState.hyperparameters === 'object'
          ? {
              ...getDefaultHyperparameters(nextTaskType, nextModelType),
              ...restoredState.hyperparameters,
            }
          : getDefaultHyperparameters(nextTaskType, nextModelType)
        : shouldResetModel
          ? getDefaultHyperparameters(nextTaskType, DEFAULT_MODEL_TYPE)
          : hyperparameters,
    )

    if (restoredState) {
      setProcessForm({
        ...getDefaultProcessForm(nextAction),
        ...(restoredState?.processForm && typeof restoredState.processForm === 'object'
          ? restoredState.processForm
          : {}),
        action: nextAction,
        columns: Array.isArray(restoredState?.processForm?.columns)
          ? restoredState.processForm.columns.filter((column) => datasetColumns.includes(column))
          : [],
      })
    }

    setAnalysisForm((previous) => ({
      ...((rawAnalysisForm || restoredState)
        ? {
            ...getDefaultAnalysisForm(nextAnalysisType),
            ...(rawAnalysisForm ?? {}),
            analysis_type: nextAnalysisType,
          }
        : previous),
      columns: (
        Array.isArray(rawAnalysisForm?.columns)
          ? rawAnalysisForm.columns
          : restoredState
            ? []
            : previous.columns
      ).filter((column) => datasetColumns.includes(column)),
      feature_columns: (
        Array.isArray(rawAnalysisForm?.feature_columns)
          ? rawAnalysisForm.feature_columns
          : restoredState
            ? []
            : previous.feature_columns
      ).filter((column) => datasetColumns.includes(column)),
      target_column: datasetColumns.includes(
        rawAnalysisForm ? rawAnalysisForm.target_column : restoredState ? '' : previous.target_column,
      )
        ? rawAnalysisForm
          ? rawAnalysisForm.target_column
          : restoredState
            ? ''
            : previous.target_column
        : '',
      group_column: datasetColumns.includes(
        rawAnalysisForm ? rawAnalysisForm.group_column : restoredState ? '' : previous.group_column,
      )
        ? rawAnalysisForm
          ? rawAnalysisForm.group_column
          : restoredState
            ? ''
            : previous.group_column
        : '',
      row_column: datasetColumns.includes(
        rawAnalysisForm ? rawAnalysisForm.row_column : restoredState ? '' : previous.row_column,
      )
        ? rawAnalysisForm
          ? rawAnalysisForm.row_column
          : restoredState
            ? ''
            : previous.row_column
        : '',
      column_column: datasetColumns.includes(
        rawAnalysisForm ? rawAnalysisForm.column_column : restoredState ? '' : previous.column_column,
      )
        ? rawAnalysisForm
          ? rawAnalysisForm.column_column
          : restoredState
            ? ''
            : previous.column_column
        : '',
    }))
  }

  const handleProcessFormChange = (field, value) => {
    if (field === 'action') {
      setProcessForm(getDefaultProcessForm(value))
      setProcessError(null)
      return
    }
    setProcessForm((previous) => ({ ...previous, [field]: value }))
    setProcessError(null)
  }

  const handleAnalysisFormChange = (field, value) => {
    if (field === 'analysis_type') {
      setAnalysisForm(getDefaultAnalysisForm(value))
      setAnalysisError(null)
      return
    }
    setAnalysisForm((previous) => ({ ...previous, [field]: value }))
    setAnalysisError(null)
  }

  const handleLoadProcessPreset = (preset) => {
    if (!preset || typeof preset !== 'object') return

    const nextAction = typeof preset.action === 'string' ? preset.action : 'handle_missing'
    const datasetColumns = Array.isArray(dataInfo?.columns) ? dataInfo.columns : []
    setProcessForm({
      ...getDefaultProcessForm(nextAction),
      ...preset,
      action: nextAction,
      columns: Array.isArray(preset.columns)
        ? preset.columns.filter((column) => datasetColumns.includes(column))
        : [],
    })
    setProcessError(null)
    setActiveView('data_prepare')
  }

  const handleFileChange = (event) => {
    if (event.target.files?.[0]) {
      setFile(event.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!file) return
    if (!(await ensureBackendReady('上传数据集'))) return

    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)
    formData.append('project_name', projectName)

    try {
      const response = await axios.post(`${API_BASE}/api/data/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const uploadedData = extractDatasetPayload(response.data)
      if (!uploadedData) {
        throw new Error('上传后的数据结构无效，请检查返回结果。')
      }
      setDataInfo(uploadedData)
      await refreshPlatformState()
      syncDatasetToModeling(uploadedData)
      clearDerivedOutputs()
      setActiveView('data_prepare')
    } catch (error) {
      const message = isNetworkError(error)
        ? handleBackendUnavailable('上传数据集')
        : `上传失败：${formatErrorMessage(error.response?.data?.detail || error.message)}`
      window.alert(message)
    } finally {
      setLoading(false)
    }
  }

  const handleApplyProcess = async () => {
    if (!dataInfo) return
    if (!(await ensureBackendReady('数据处理'))) return

    setProcessLoading(true)
    setProcessError(null)

    try {
      const response = await axios.post(`${API_BASE}/api/data/process`, processForm)
      const nextDataset = extractDatasetPayload(response.data)
      if (!nextDataset) {
        throw new Error('处理后的数据结构无效，请检查返回结果。')
      }
      setDataInfo(nextDataset)
      syncDatasetToModeling(nextDataset)
      clearDerivedOutputs()
    } catch (error) {
      setProcessError(
        isNetworkError(error)
          ? handleBackendUnavailable('数据处理')
          : formatErrorMessage(error.response?.data?.detail || error.message),
      )
    } finally {
      setProcessLoading(false)
    }
  }

  const handleTargetChange = (event) => {
    const nextTarget = event.target.value
    const datasetColumns = Array.isArray(dataInfo?.columns) ? dataInfo.columns : []
    const nextTaskType = inferTaskType(dataInfo, nextTarget)
    const nextAvailableFeatures = getAvailableFeatureColumns(dataInfo, nextTarget)

    setTargetCol(nextTarget)
    setSelectedFeatures((previous) => {
      const filtered = previous.filter(
        (feature) => feature !== nextTarget && datasetColumns.includes(feature),
      )
      return filtered.length > 0 ? filtered : nextAvailableFeatures
    })
    setTaskType(nextTaskType)
    setModelType(DEFAULT_MODEL_TYPE)
    setHyperparameters(getDefaultHyperparameters(nextTaskType, DEFAULT_MODEL_TYPE))
    clearTrainingState()
    clearReportState()
  }

  const handleFeatureChange = (event) => {
    const value = event.target.value
    const nextFeatures = typeof value === 'string' ? value.split(',') : value
    setSelectedFeatures(nextFeatures.filter((feature) => feature !== targetCol))
    clearTrainingState()
    clearReportState()
  }

  const handleTaskTypeChange = (event) => {
    const nextTaskType = event.target.value
    setTaskType(nextTaskType)
    setModelType(DEFAULT_MODEL_TYPE)
    setHyperparameters(getDefaultHyperparameters(nextTaskType, DEFAULT_MODEL_TYPE))
    clearTrainingState()
    clearReportState()
  }

  const handleModelTypeChange = (event) => {
    const nextModelType = event.target.value
    setModelType(nextModelType)
    setHyperparameters(getDefaultHyperparameters(taskType, nextModelType))
    clearTrainingState()
    clearReportState()
  }

  const handleHyperparameterChange = (name, value) => {
    setHyperparameters((previous) => ({ ...previous, [name]: value }))
    clearTrainingState()
    clearReportState()
  }

  const handleTrain = async () => {
    if (!targetCol || selectedFeatures.length === 0) {
      setTrainError('请选择 1 个目标列，并至少选择 1 个特征列。')
      setTrainingStatus('error')
      return
    }
    if (!(await ensureBackendReady('模型训练'))) return

    setTrainingStatus('training')
    setTrainError(null)
    setTrainingResult(null)
    clearReportState()

    try {
      const response = await axios.post(`${API_BASE}/api/model/train`, {
        project_name: projectName,
        target_column: targetCol,
        feature_columns: selectedFeatures,
        task_type: taskType,
        model_type: modelType,
        hyperparameters,
      })
      setTrainingResult(response.data)
      await refreshPlatformState()
      setTrainingStatus('success')
      setActiveView('modeling')
    } catch (error) {
      setTrainError(
        isNetworkError(error)
          ? handleBackendUnavailable('模型训练')
          : formatErrorMessage(error.response?.data?.detail || error.message),
      )
      setTrainingStatus('error')
    }
  }

  const handleRunAnalysis = async () => {
    if (!dataInfo) return
    if (!(await ensureBackendReady('统计分析'))) return

    setAnalysisStatus('loading')
    setAnalysisError(null)
    clearReportState()

    try {
      const response = await axios.post(`${API_BASE}/api/analysis/run`, {
        ...analysisForm,
        project_name: projectName,
      })
      const nextResult = normalizeAnalysisResult(response.data)
      if (!nextResult) {
        throw new Error('统计分析结果为空。')
      }
      setAnalysisResults((previous) => [...previous, nextResult])
      await refreshPlatformState()
      setAnalysisStatus('success')
      setActiveView('modeling')
    } catch (error) {
      setAnalysisError(
        isNetworkError(error)
          ? handleBackendUnavailable('统计分析')
          : formatErrorMessage(error.response?.data?.detail || error.message),
      )
      setAnalysisStatus('error')
    }
  }

  const handleGenerateReport = async () => {
    if (!dataInfo) return
    if (!(await ensureBackendReady('生成报告'))) return

    setReportStatus('loading')
    setReportError(null)

    try {
      const response = await axios.post(`${API_BASE}/api/report/generate`, {
        project_name: projectName,
        dataset: dataInfo,
        processing_history: dataInfo.history ?? [],
        training_result: trainingResult ?? {},
        analysis_results: analysisResults,
      })
      const nextReport = normalizeReportResult(response.data)
      if (!nextReport) {
        throw new Error('报告结果为空。')
      }
      setReportResult(nextReport)
      await refreshPlatformState()
      setReportStatus('success')
      setActiveView('report')
    } catch (error) {
      setReportError(
        isNetworkError(error)
          ? handleBackendUnavailable('生成报告')
          : formatErrorMessage(error.response?.data?.detail || error.message),
      )
      setReportStatus('error')
    }
  }

  const handleSaveSnapshot = async () => {
    if (!(await ensureBackendReady('保存项目'))) return

    const localSnapshot = {
      projectName,
      projectNotes,
      savedAt: new Date().toISOString(),
      dataset: dataInfo
        ? {
            filename: dataInfo.filename,
            shape: dataInfo.shape,
            columns: dataInfo.columns,
          }
        : null,
      processingHistory: dataInfo?.history ?? [],
      trainingResult,
      analysisResults,
      reportTitle: reportResult?.title ?? null,
      workspaceState: {
        activeView,
        targetCol,
        selectedFeatures,
        taskType,
        modelType,
        hyperparameters,
        processForm,
        analysisForm,
      },
    }
    const backendSnapshot = {
      project_name: projectName,
      project_notes: projectNotes,
      dataset: dataInfo ?? {},
      processing_history: dataInfo?.history ?? [],
      training_result: trainingResult ?? {},
      analysis_results: analysisResults,
      report_result: reportResult ?? {},
      workspace_state: {
        activeView,
        targetCol,
        selectedFeatures,
        taskType,
        modelType,
        hyperparameters,
        processForm,
        analysisForm,
      },
    }

    setProjectSaveStatus('saving')
    setProjectSaveError(null)

    try {
      window.localStorage.setItem('ml_studio_snapshot', JSON.stringify(localSnapshot))
      const response = await axios.post(`${API_BASE}/api/project/save`, backendSnapshot)
      setSavedProjectInfo(response.data)
      await refreshPlatformState()
      setProjectSaveStatus('success')
    } catch (error) {
      setProjectSaveError(
        isNetworkError(error)
          ? handleBackendUnavailable('保存项目')
          : formatErrorMessage(error.response?.data?.detail || error.message),
      )
      setProjectSaveStatus('error')
    }
  }

  const handleOpenProject = async (projectId) => {
    if (!(await ensureBackendReady('打开项目'))) return

    setProjectLoadStatus('loading')
    setProjectLoadError(null)
    setLoadingProjectId(projectId)

    try {
      const response = await axios.get(`${API_BASE}/api/projects/${projectId}`)
      const project = response.data?.project ?? null
      const snapshot = response.data?.snapshot?.data ?? {}
      const workspaceState =
        snapshot.workspace_state && typeof snapshot.workspace_state === 'object'
          ? snapshot.workspace_state
          : {}
      const restoredDataset = extractDatasetPayload(response.data?.dataset)
      const restoredTrainingResult = normalizeTrainingResult(snapshot.training_result)
      const restoredAnalysisResults = Array.isArray(snapshot.analysis_results)
        ? snapshot.analysis_results
            .map((item) => normalizeAnalysisResult(item))
            .filter(Boolean)
        : []
      const restoredReportResult =
        snapshot.report_result && Object.keys(snapshot.report_result).length > 0
          ? normalizeReportResult(snapshot.report_result)
          : null

      if (project?.name) {
        setProjectName(project.name)
      }
      setProjectNotes(snapshot.project_notes ?? project?.notes ?? '')
      setFile(null)
      setProcessError(null)

      if (restoredDataset) {
        setDataInfo(restoredDataset)
        syncDatasetToModeling(restoredDataset, {
          ...workspaceState,
          targetCol: workspaceState.targetCol ?? restoredTrainingResult?.target_column,
          selectedFeatures:
            workspaceState.selectedFeatures ?? restoredTrainingResult?.feature_columns ?? [],
          taskType: workspaceState.taskType ?? restoredTrainingResult?.task_type,
          modelType:
            workspaceState.modelType ??
            restoredTrainingResult?.model_type ??
            DEFAULT_MODEL_TYPE,
          hyperparameters:
            workspaceState.hyperparameters ??
            restoredTrainingResult?.hyperparameters ??
            {},
        })
      } else {
        setDataInfo(null)
        setProcessForm(getDefaultProcessForm())
        setAnalysisForm(getDefaultAnalysisForm())
      }

      setTrainingResult(restoredTrainingResult)
      setTrainingStatus(restoredTrainingResult ? 'success' : 'idle')
      setTrainError(null)

      setAnalysisResults(restoredAnalysisResults)
      setAnalysisStatus(restoredAnalysisResults.length > 0 ? 'success' : 'idle')
      setAnalysisError(null)

      setReportResult(restoredReportResult)
      setReportStatus(restoredReportResult ? 'success' : 'idle')
      setReportError(null)

      setSavedProjectInfo({ project })
      await refreshPlatformState()
      const savedView = VIEW_ITEMS.some((item) => item.value === workspaceState.activeView)
        ? workspaceState.activeView
        : restoredDataset
          ? 'data_prepare'
          : 'project'
      setActiveView(
        savedView === 'data_prepare' || savedView === 'modeling'
          ? restoredDataset
            ? savedView
            : 'project'
          : savedView,
      )
      setProjectLoadStatus('success')
    } catch (error) {
      setProjectLoadError(
        isNetworkError(error)
          ? handleBackendUnavailable('打开项目')
          : formatErrorMessage(error.response?.data?.detail || error.message),
      )
      setProjectLoadStatus('error')
    } finally {
      setLoadingProjectId(null)
    }
  }

  const isTrainDisabled =
    !dataInfo ||
    !targetCol ||
    selectedFeatures.length === 0 ||
    trainingStatus === 'training' ||
    !isBackendReady

  return (
    <Container
      maxWidth={false}
      sx={{
        maxWidth: 1720,
        mx: 'auto',
        px: { xs: 2, md: 3.5, lg: 4.5 },
        py: { xs: 3, md: 4 },
      }}
    >
      <Stack spacing={3}>
        {!isBackendReady && (
          <Alert
            severity={backendStatus === 'error' ? 'error' : 'info'}
            action={
              <Button color="inherit" size="small" onClick={handleRetryBackend}>
                重新检测
              </Button>
            }
            sx={{ borderRadius: RADII.card }}
          >
            {backendStatusMessage}
          </Alert>
        )}

        <Paper
          sx={{
            p: { xs: 3, md: 4 },
            borderRadius: RADII.hero,
            border: '1px solid',
            borderColor: 'divider',
            background:
              'linear-gradient(135deg, rgba(217, 119, 87, 0.14) 0%, rgba(255, 252, 246, 0.98) 48%, rgba(106, 155, 204, 0.10) 100%)',
          }}
        >
          <Stack spacing={2.25}>
            <Chip
              label="ML Studio"
              sx={{
                alignSelf: 'flex-start',
                fontFamily: '"Noto Sans SC", "Microsoft YaHei", sans-serif',
                fontWeight: 700,
                color: 'primary.main',
                backgroundColor: 'rgba(217, 119, 87, 0.12)',
              }}
            />
            <Typography variant="h3" component="h1" sx={{ maxWidth: 980 }}>
              机器学习与统计分析平台
            </Typography>
            <Typography
              variant="body1"
              sx={{ maxWidth: 1120, color: 'text.secondary', lineHeight: 1.85 }}
            >
              当前版本已经从单页训练器扩展成四段式工作台，覆盖数据准备、分析建模、结果报告和项目管理，
              更接近你想要的桌面分析平台形态。
            </Typography>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.25} useFlexGap flexWrap="wrap">
              <StatChip
                icon={<CloudUploadIcon fontSize="small" />}
                label="数据准备"
                value="缺失值 / 异常值 / 编码 / 派生变量"
              />
              <StatChip
                icon={<PsychologyIcon fontSize="small" />}
                label="分析建模"
                value="机器学习训练 + 统计分析"
              />
              <StatChip
                icon={<InsightsIcon fontSize="small" />}
                label="结果报告"
                value="自动摘要 / Word / PDF"
              />
            </Stack>
          </Stack>
        </Paper>

        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', md: '260px minmax(0, 1fr)' },
            gap: 3,
            alignItems: 'start',
          }}
        >
          <Paper
            sx={{
              width: '100%',
              p: 1.5,
              borderRadius: RADII.card,
              position: { md: 'sticky' },
              top: { md: 24 },
              maxHeight: { md: 'calc(100svh - 48px)' },
              overflowY: { md: 'auto' },
            }}
          >
            <Stack spacing={1}>
              <Typography variant="h6" sx={{ px: 1, pt: 0.5 }}>
                工作区导航
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ px: 1, lineHeight: 1.75 }}>
                桌面端默认采用左侧工作区切换，让数据准备、建模分析、报告与项目管理保持清晰层次。
              </Typography>
              {VIEW_ITEMS.map((item) => {
                const selected = activeView === item.value
                return (
                  <Button
                    key={item.value}
                    variant={selected ? 'contained' : 'text'}
                    color={selected ? 'primary' : 'inherit'}
                    startIcon={item.icon}
                    onClick={() => setActiveView(item.value)}
                    sx={{
                      justifyContent: 'flex-start',
                      py: 1.25,
                      px: 1.5,
                      borderRadius: RADII.inset,
                      color: selected ? 'primary.contrastText' : 'text.primary',
                      backgroundColor: selected ? undefined : 'rgba(255, 252, 246, 0.76)',
                      border: selected ? 'none' : '1px solid rgba(232, 230, 220, 0.9)',
                    }}
                  >
                    {item.label}
                  </Button>
                )
              })}
            </Stack>

            <Paper
              variant="outlined"
              sx={{
                mt: 2,
                p: 1.5,
                borderRadius: RADII.inset,
                backgroundColor: 'rgba(255, 252, 246, 0.84)',
              }}
            >
              <Stack spacing={0.75}>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>
                  当前状态
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  本地服务：{isBackendReady ? '已连接' : backendStatus === 'starting' ? '启动中' : '不可用'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  数据集：{dataInfo?.filename ?? '未上传'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  处理步骤：{dataInfo?.history?.length ?? 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  分析结果：{analysisResults.length}
                </Typography>
              </Stack>
            </Paper>
          </Paper>

          <Stack spacing={3} sx={{ minWidth: 0 }}>
            {activeView === 'data_prepare' && (
              <DataPrepWorkspace
                file={file}
                loading={loading}
                onFileChange={handleFileChange}
                onUpload={handleUpload}
                dataInfo={dataInfo}
                taskType={taskType}
                processForm={processForm}
                onProcessFormChange={handleProcessFormChange}
                onApplyProcess={handleApplyProcess}
                processLoading={processLoading}
                processError={processError}
                previewTableMinWidth={previewTableMinWidth}
                onLoadProcessPreset={handleLoadProcessPreset}
              />
            )}

            {activeView === 'modeling' && (
              <ModelingWorkspace
                dataInfo={dataInfo}
                targetCol={targetCol}
                selectedFeatures={selectedFeatures}
                taskType={taskType}
                modelType={modelType}
                hyperparameters={hyperparameters}
                availableFeatureColumns={availableFeatureColumns}
                onTargetChange={handleTargetChange}
                onFeatureChange={handleFeatureChange}
                onTaskTypeChange={handleTaskTypeChange}
                onModelTypeChange={handleModelTypeChange}
                onHyperparameterChange={handleHyperparameterChange}
                onTrain={handleTrain}
                isTrainDisabled={isTrainDisabled}
                trainingStatus={trainingStatus}
                trainingResult={trainingResult}
                trainError={trainError}
                analysisProps={{
                  dataInfo,
                  analysisForm,
                  onAnalysisFormChange: handleAnalysisFormChange,
                  onRunAnalysis: handleRunAnalysis,
                  analysisStatus,
                  analysisError,
                  analysisResults,
                }}
              />
            )}

            {activeView === 'report' && (
              <ReportWorkspace
                projectName={projectName}
                onProjectNameChange={setProjectName}
                dataInfo={dataInfo}
                processingHistory={dataInfo?.history ?? []}
                trainingResult={trainingResult}
                analysisResults={analysisResults}
                reportStatus={reportStatus}
                reportError={reportError}
                reportResult={reportResult}
                onGenerateReport={handleGenerateReport}
              />
            )}

            {activeView === 'project' && (
              <ProjectWorkspace
                projectName={projectName}
                projectNotes={projectNotes}
                onProjectNameChange={setProjectName}
                onProjectNotesChange={setProjectNotes}
                dataInfo={dataInfo}
                processingHistory={dataInfo?.history ?? []}
                trainingResult={trainingResult}
                analysisResults={analysisResults}
                reportResult={reportResult}
                onSaveSnapshot={handleSaveSnapshot}
                saveStatus={projectSaveStatus}
                saveError={projectSaveError}
                savedProjectInfo={savedProjectInfo}
                loadStatus={projectLoadStatus}
                loadError={projectLoadError}
                loadingProjectId={loadingProjectId}
                onOpenProject={handleOpenProject}
                recentProjects={recentProjects}
              />
            )}
          </Stack>
        </Box>
      </Stack>
    </Container>
  )
}

export default App
