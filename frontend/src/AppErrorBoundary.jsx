import { Component } from 'react'
import { Alert, Box, Button, Paper, Stack, Typography } from '@mui/material'

export default class AppErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, errorMessage: '' }
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      errorMessage: error?.message || '页面发生了运行时错误。',
    }
  }

  componentDidCatch(error, info) {
    console.error('App runtime error:', error, info)
  }

  handleReload = () => {
    window.location.reload()
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children
    }

    return (
      <Box sx={{ minHeight: '100svh', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 3 }}>
        <Paper sx={{ maxWidth: 760, width: '100%', p: 4 }}>
          <Stack spacing={2}>
            <Typography variant="h4">页面渲染失败</Typography>
            <Alert severity="error">{this.state.errorMessage}</Alert>
            <Typography color="text.secondary">
              我已经加了错误兜底。如果你再次看到这个页面，把上面的错误信息发给我，我可以继续精准修复。
            </Typography>
            <Button variant="contained" onClick={this.handleReload}>
              重新加载页面
            </Button>
          </Stack>
        </Paper>
      </Box>
    )
  }
}
