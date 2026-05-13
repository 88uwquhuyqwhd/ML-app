import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import App from './App.jsx'
import AppErrorBoundary from './AppErrorBoundary.jsx'
import './index.css'

const theme = createTheme({
  shape: {
    borderRadius: 10,
  },
  palette: {
    mode: 'light',
    primary: {
      main: '#d97757',
      light: '#efc2b2',
      dark: '#b96349',
      contrastText: '#faf9f5',
    },
    secondary: {
      main: '#6a9bcc',
    },
    success: {
      main: '#788c5d',
    },
    warning: {
      main: '#b57b35',
    },
    text: {
      primary: '#141413',
      secondary: '#6f6b63',
    },
    background: {
      default: '#faf9f5',
      paper: '#fffdf8',
    },
    divider: '#e8e6dc',
  },
  typography: {
    fontFamily:
      '"Noto Serif SC", "Source Han Serif SC", "Songti SC", "Microsoft YaHei", serif',
    h1: {
      fontFamily:
        '"Noto Sans SC", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif',
      fontWeight: 600,
      letterSpacing: '-0.04em',
    },
    h2: {
      fontFamily:
        '"Noto Sans SC", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif',
      fontWeight: 600,
      letterSpacing: '-0.03em',
    },
    h3: {
      fontFamily:
        '"Noto Sans SC", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif',
      fontWeight: 600,
      letterSpacing: '-0.03em',
    },
    h4: {
      fontFamily:
        '"Noto Sans SC", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif',
      fontWeight: 600,
      letterSpacing: '-0.02em',
    },
    h5: {
      fontFamily:
        '"Noto Sans SC", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif',
      fontWeight: 600,
    },
    h6: {
      fontFamily:
        '"Noto Sans SC", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif',
      fontWeight: 600,
    },
    button: {
      fontFamily:
        '"Noto Sans SC", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif',
      fontWeight: 600,
      textTransform: 'none',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background:
            'radial-gradient(circle at top left, rgba(217, 119, 87, 0.10), transparent 24%), radial-gradient(circle at top right, rgba(106, 155, 204, 0.08), transparent 22%), linear-gradient(180deg, #fdfbf7 0%, #f8f4ec 100%)',
          color: '#141413',
        },
        '::selection': {
          backgroundColor: 'rgba(217, 119, 87, 0.22)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 14,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          border: '1px solid #e8e6dc',
          borderRadius: 14,
          boxShadow: '0 12px 32px rgba(20, 20, 19, 0.05)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          paddingInline: 20,
          boxShadow: 'none',
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          backgroundColor: 'rgba(255, 253, 248, 0.84)',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          fontFamily: '"Poppins", Arial, sans-serif',
          fontWeight: 600,
          color: '#141413',
          backgroundColor: '#f4f1e8',
        },
      },
    },
  },
})

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppErrorBoundary>
        <App />
      </AppErrorBoundary>
    </ThemeProvider>
  </StrictMode>,
)
