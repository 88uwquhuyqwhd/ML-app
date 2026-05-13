const { app, BrowserWindow, dialog } = require('electron')
const fs = require('fs')
const path = require('path')
const { pathToFileURL } = require('url')
const { spawn } = require('child_process')

let mainWindow
let pythonProcess

const PORT = 8010

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1120,
    minHeight: 760,
    title: 'ML Pro Studio',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  })

  const startUrl = app.isPackaged
    ? pathToFileURL(path.join(__dirname, '../dist/index.html')).toString()
    : process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173'
  mainWindow.loadURL(startUrl)

  if (process.env.ML_APP_OPEN_DEVTOOLS === '1') {
    mainWindow.webContents.openDevTools({ mode: 'detach' })
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

function getProductionBackendEnv() {
  const runtimeRoot = path.join(app.getPath('userData'), 'runtime')
  const backendDataRoot = path.join(runtimeRoot, 'backend')

  fs.mkdirSync(backendDataRoot, { recursive: true })

  return {
    ...process.env,
    ML_APP_PORT: String(PORT),
    ML_APP_DATA_ROOT: backendDataRoot,
    ML_APP_DB_PATH: path.join(backendDataRoot, 'ml_studio.db'),
    ML_APP_UPLOAD_DIR: path.join(backendDataRoot, 'uploads'),
    ML_APP_PROJECT_STATE_DIR: path.join(backendDataRoot, 'project_state'),
  }
}

function startPythonBackend() {
  const isDev = !app.isPackaged

  if (isDev) {
    const pythonExePath = path.join(__dirname, '../../backend/venv/Scripts/python.exe')
    const apiScriptPath = path.join(__dirname, '../../backend/main.py')

    console.log(`Starting Python backend from: ${apiScriptPath}`)
    pythonProcess = spawn(
      pythonExePath,
      ['-m', 'uvicorn', 'main:app', '--reload', '--host', '127.0.0.1', '--port', String(PORT)],
      {
        cwd: path.join(__dirname, '../../backend'),
        env: {
          ...process.env,
          ML_APP_PORT: String(PORT),
        },
        windowsHide: true,
      },
    )
  } else {
    const backendExecutable = path.join(process.resourcesPath, 'api', 'api.exe')

    if (!fs.existsSync(backendExecutable)) {
      const message = `Packaged backend not found: ${backendExecutable}`
      console.error(message)
      dialog.showErrorBox('Backend Startup Failed', message)
      return false
    }

    console.log(`Starting packaged backend from: ${backendExecutable}`)
    pythonProcess = spawn(backendExecutable, [], {
      cwd: process.resourcesPath,
      env: getProductionBackendEnv(),
      windowsHide: true,
    })
  }

  if (!pythonProcess) {
    return false
  }

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Backend stdout: ${data}`)
  })

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Backend stderr: ${data}`)
  })

  pythonProcess.on('error', (error) => {
    console.error('Backend process failed to start:', error)
    dialog.showErrorBox(
      'Backend Startup Failed',
      `Unable to start the bundled backend.\n\n${error.message}`,
    )
  })

  pythonProcess.on('close', (code) => {
    console.log(`Python backend process exited with code ${code}`)
  })

  return true
}

function stopPythonBackend() {
  if (!pythonProcess?.pid) {
    return
  }

  console.log('Killing Python backend process...')

  if (process.platform === 'win32') {
    spawn('taskkill', ['/pid', String(pythonProcess.pid), '/f', '/t'], { windowsHide: true })
  } else {
    pythonProcess.kill()
  }

  pythonProcess = null
}

app.whenReady().then(() => {
  const backendStarted = startPythonBackend()
  if (backendStarted === false) {
    app.quit()
    return
  }
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('will-quit', () => {
  stopPythonBackend()
})
