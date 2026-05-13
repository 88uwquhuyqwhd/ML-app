import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import FolderCopyIcon from '@mui/icons-material/FolderCopy'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import SaveIcon from '@mui/icons-material/Save'
import { RADII, SectionHeader, getSafeShape } from './studioShared.jsx'

export default function ProjectWorkspace({
  projectName,
  projectNotes,
  onProjectNameChange,
  onProjectNotesChange,
  dataInfo,
  processingHistory,
  trainingResult,
  analysisResults,
  reportResult,
  onSaveSnapshot,
  saveStatus,
  saveError,
  savedProjectInfo,
  loadStatus,
  loadError,
  loadingProjectId,
  onOpenProject,
  recentProjects,
}) {
  const shape = dataInfo ? getSafeShape(dataInfo) : [0, 0]
  const latestSavedProject = savedProjectInfo?.project ?? null
  const projectItems = Array.isArray(recentProjects) ? recentProjects : []

  return (
    <Stack spacing={3}>
      {loadError && <Alert severity="error">{loadError}</Alert>}
      {loadStatus === 'success' && !loadError && (
        <Alert severity="success">
          已恢复当前项目工作区，数据、训练结果和报告状态已同步到页面。
        </Alert>
      )}

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', xl: 'minmax(360px, 0.95fr) minmax(0, 1.05fr)' },
          gap: 3,
          alignItems: 'start',
        }}
      >
        <Card sx={{ borderRadius: RADII.card }}>
          <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
            <Stack spacing={2.25}>
              <SectionHeader
                chip="项目管理"
                title="项目信息与快照"
                description="保存当前工作区状态，记录业务背景、字段说明和阶段结论，方便后续继续分析。"
                icon={<FolderCopyIcon fontSize="small" />}
              />

              <TextField
                fullWidth
                label="项目名称"
                value={projectName}
                onChange={(event) => onProjectNameChange(event.target.value)}
              />
              <TextField
                fullWidth
                multiline
                minRows={4}
                label="项目备注"
                value={projectNotes}
                onChange={(event) => onProjectNotesChange(event.target.value)}
                helperText="建议记录业务背景、变量说明、建模假设和当前结论。"
              />

              <Button
                variant="contained"
                size="large"
                startIcon={saveStatus === 'saving' ? undefined : <SaveIcon />}
                onClick={onSaveSnapshot}
                disabled={saveStatus === 'saving'}
              >
                {saveStatus === 'saving' ? <CircularProgress size={22} color="inherit" /> : '保存当前项目快照'}
              </Button>

              {saveError && <Alert severity="error">{saveError}</Alert>}

              {saveStatus === 'success' && latestSavedProject && (
                <Alert severity="success">
                  项目已保存，可在“最近项目”中重新打开。最近更新时间：{latestSavedProject.updated_at}
                </Alert>
              )}
            </Stack>
          </CardContent>
        </Card>

        <Card sx={{ borderRadius: RADII.card }}>
          <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
            <Stack spacing={2}>
              <Typography variant="h5">当前工作区概览</Typography>
              <Paper variant="outlined" sx={{ p: 2, borderRadius: RADII.inset }}>
                <Stack spacing={1}>
                  <Typography variant="body2">数据集：{dataInfo?.filename ?? '未上传'}</Typography>
                  <Typography variant="body2">
                    数据规模：{dataInfo ? `${shape[0]} 行 / ${shape[1]} 列` : '暂无'}
                  </Typography>
                  <Typography variant="body2">数据处理步骤：{processingHistory.length}</Typography>
                  <Typography variant="body2">
                    机器学习结果：{trainingResult ? trainingResult.message : '暂无'}
                  </Typography>
                  <Typography variant="body2">统计分析结果：{analysisResults.length} 项</Typography>
                  <Typography variant="body2">
                    报告状态：{reportResult ? `已生成 · ${reportResult.generated_at}` : '未生成'}
                  </Typography>
                </Stack>
              </Paper>
            </Stack>
          </CardContent>
        </Card>
      </Box>

      <Card sx={{ borderRadius: RADII.card }}>
        <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
          <Stack spacing={1.5}>
            <SectionHeader
              chip="项目仓库"
              title="最近项目"
              description="这里会列出本地最近保存或更新的项目，你可以直接恢复到当前工作区。"
              icon={<FolderOpenIcon fontSize="small" />}
            />

            {projectItems.length ? (
              <Stack spacing={1.25}>
                {projectItems.map((item) => {
                  const isCurrentProject = item.name === projectName
                  const isLoading = loadingProjectId === item.id

                  return (
                    <Paper
                      key={`${item.id}-${item.updated_at}`}
                      variant="outlined"
                      sx={{
                        p: 1.5,
                        borderRadius: RADII.inset,
                        borderColor: isCurrentProject ? 'primary.main' : 'divider',
                        backgroundColor: isCurrentProject
                          ? 'rgba(217, 119, 87, 0.07)'
                          : 'rgba(255, 252, 246, 0.88)',
                      }}
                    >
                      <Stack
                        direction={{ xs: 'column', md: 'row' }}
                        spacing={1.5}
                        justifyContent="space-between"
                        alignItems={{ xs: 'stretch', md: 'center' }}
                      >
                        <Stack spacing={0.5} sx={{ minWidth: 0 }}>
                          <Typography variant="body2" sx={{ fontWeight: 700 }}>
                            {item.name}
                          </Typography>
                          {item.notes ? (
                            <Typography
                              variant="body2"
                              color="text.secondary"
                              sx={{
                                lineHeight: 1.75,
                                display: '-webkit-box',
                                WebkitBoxOrient: 'vertical',
                                WebkitLineClamp: 2,
                                overflow: 'hidden',
                              }}
                            >
                              {item.notes}
                            </Typography>
                          ) : null}
                          <Typography variant="caption" color="text.secondary">
                            最近更新：{item.updated_at}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            数据集：{item.latest_dataset_filename || '暂无'} | 规模：{item.latest_row_count ?? 0} 行 /{' '}
                            {item.latest_column_count ?? 0} 列
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            快照 {item.snapshot_count ?? 0} | 训练 {item.training_run_count ?? 0} | 分析{' '}
                            {item.analysis_run_count ?? 0} | 报告 {item.report_count ?? 0}
                          </Typography>
                        </Stack>

                        <Button
                          variant={isCurrentProject ? 'contained' : 'outlined'}
                          onClick={() => onOpenProject?.(item.id)}
                          disabled={!onOpenProject || isLoading}
                          startIcon={isLoading ? undefined : <FolderOpenIcon />}
                          sx={{ alignSelf: { xs: 'stretch', md: 'center' } }}
                        >
                          {isLoading ? (
                            <CircularProgress size={20} color="inherit" />
                          ) : isCurrentProject ? (
                            '重新加载'
                          ) : (
                            '打开项目'
                          )}
                        </Button>
                      </Stack>
                    </Paper>
                  )
                })}
              </Stack>
            ) : (
              <Alert severity="info">
                还没有项目历史。保存一次项目快照后，这里就会形成你的项目列表。
              </Alert>
            )}
          </Stack>
        </CardContent>
      </Card>

      {processingHistory.length > 0 ? (
        <Card sx={{ borderRadius: RADII.card }}>
          <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
            <Typography variant="h6" sx={{ mb: 1.25 }}>
              最近处理记录
            </Typography>
            <Stack spacing={1}>
              {processingHistory
                .slice()
                .reverse()
                .slice(0, 8)
                .map((item, index) => (
                  <Paper
                    key={`${item.created_at}-${index}`}
                    variant="outlined"
                    sx={{ p: 1.5, borderRadius: RADII.inset, backgroundColor: 'rgba(255, 252, 246, 0.9)' }}
                  >
                    <Typography variant="body2" sx={{ fontWeight: 700, mb: 0.5 }}>
                      {index + 1}. {item.label}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {item.detail}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {item.created_at}
                    </Typography>
                  </Paper>
                ))}
            </Stack>
          </CardContent>
        </Card>
      ) : (
        <Alert severity="info">
          暂时还没有处理记录，项目快照会在你开始操作后更有价值。
        </Alert>
      )}
    </Stack>
  )
}
