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
import ArticleIcon from '@mui/icons-material/Article'
import DescriptionIcon from '@mui/icons-material/Description'
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf'
import { RADII, SectionHeader } from './studioShared.jsx'

function downloadBlob(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export default function ReportWorkspace({
  projectName,
  onProjectNameChange,
  dataInfo,
  processingHistory,
  trainingResult,
  analysisResults,
  reportStatus,
  reportError,
  reportResult,
  onGenerateReport,
}) {
  const canGenerate = Boolean(dataInfo && (trainingResult || analysisResults.length))
  const sections = Array.isArray(reportResult?.sections) ? reportResult.sections : []

  const handleExportWord = () => {
    if (!reportResult?.html) return
    downloadBlob(`${projectName || '分析报告'}.doc`, reportResult.html, 'application/msword')
  }

  const handleExportMarkdown = () => {
    if (!reportResult?.markdown) return
    downloadBlob(`${projectName || '分析报告'}.md`, reportResult.markdown, 'text/markdown;charset=utf-8')
  }

  const handlePrintPdf = () => {
    if (!reportResult?.html) return
    const printWindow = window.open('', '_blank', 'width=960,height=720')
    if (!printWindow) return
    printWindow.document.write(reportResult.html)
    printWindow.document.close()
    printWindow.focus()
    setTimeout(() => {
      printWindow.print()
    }, 300)
  }

  return (
    <Stack spacing={3}>
      <Card sx={{ borderRadius: RADII.card }}>
        <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
          <Stack spacing={2.25}>
            <SectionHeader
              chip="结果报告"
              title="自动报告与导出"
              description="把数据准备、训练结果和统计分析整合成报告，支持导出 Word、Markdown，并通过打印能力导出 PDF。"
              icon={<ArticleIcon fontSize="small" />}
            />

            <TextField
              fullWidth
              label="项目名称"
              value={projectName}
              onChange={(event) => onProjectNameChange(event.target.value)}
            />

            <Paper variant="outlined" sx={{ p: 1.75, borderRadius: RADII.inset }}>
              <Stack spacing={1}>
                <Typography variant="body2" color="text.secondary">
                  数据准备步骤：{processingHistory.length} 步
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  机器学习结果：{trainingResult ? '已生成' : '暂无'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  统计分析结果：{analysisResults.length} 份
                </Typography>
              </Stack>
            </Paper>

            {reportError && <Alert severity="error">{reportError}</Alert>}

            <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5}>
              <Button variant="contained" size="large" onClick={onGenerateReport} disabled={!canGenerate || reportStatus === 'loading'}>
                {reportStatus === 'loading' ? <CircularProgress size={22} color="inherit" /> : '生成报告'}
              </Button>
              <Button variant="outlined" size="large" startIcon={<DescriptionIcon />} onClick={handleExportWord} disabled={!reportResult}>
                导出 Word
              </Button>
              <Button variant="outlined" size="large" startIcon={<PictureAsPdfIcon />} onClick={handlePrintPdf} disabled={!reportResult}>
                打印 / 导出 PDF
              </Button>
              <Button variant="outlined" size="large" onClick={handleExportMarkdown} disabled={!reportResult}>
                导出 Markdown
              </Button>
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {reportResult ? (
        <Card sx={{ borderRadius: RADII.card }}>
          <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
            <Stack spacing={2.25}>
              <Typography variant="h4">{reportResult.title}</Typography>
              <Typography variant="body2" color="text.secondary">
                生成时间：{reportResult.generated_at}
              </Typography>

              {sections.map((section) => (
                <Paper key={section.title} variant="outlined" sx={{ p: 2, borderRadius: RADII.inset }}>
                  <Typography variant="h6" sx={{ mb: 0.75 }}>
                    {section.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1.25, lineHeight: 1.85 }}>
                    {section.body}
                  </Typography>
                  <Stack spacing={0.75}>
                    {(Array.isArray(section.bullets) ? section.bullets : []).map((item, index) => (
                      <Typography key={`${section.title}-${index}`} variant="body2">
                        {index + 1}. {item}
                      </Typography>
                    ))}
                  </Stack>
                </Paper>
              ))}

              <Box
                sx={{
                  p: 2,
                  borderRadius: RADII.inset,
                  border: '1px solid #e8e6dc',
                  backgroundColor: 'rgba(255, 252, 246, 0.9)',
                }}
              >
                <Typography variant="h6" sx={{ mb: 1 }}>
                  Markdown 预览
                </Typography>
                <Typography component="pre" sx={{ whiteSpace: 'pre-wrap', fontSize: 13, lineHeight: 1.7, m: 0 }}>
                  {reportResult.markdown}
                </Typography>
              </Box>
            </Stack>
          </CardContent>
        </Card>
      ) : (
        <Alert severity="info">生成报告后，这里会展示自动摘要、解读和导出入口。</Alert>
      )}
    </Stack>
  )
}
