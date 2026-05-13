# ML Studio

Windows 本地可视化机器学习与统计分析桌面应用。

它的目标不是只做一个“上传数据后点一下训练”的模型演示页，而是尽量把一条完整的分析流程串起来：

- 数据准备
- 分析建模
- 统计分析
- 结果报告
- 项目管理

适合教学演示、课程作业、业务分析、快速原型验证等场景。

## 当前能力

- 数据准备：支持 `CSV / Excel` 上传、全表预览、缺失值处理、异常值处理、编码、标准化、字段类型转换、行筛选、派生变量
- 分析建模：支持分类与回归训练、目标列与特征列配置、超参数设置、结果可视化
- 统计分析：支持描述统计、相关分析、T 检验、方差分析、卡方检验、线性回归、Logit 回归、聚类、PCA、因子分析
- 结果报告：支持自动摘要、指标解释、图表说明、Word / Markdown / PDF 导出
- 项目管理：支持项目备注、快照保存、最近项目恢复
- 演示数据：内置多场景测试数据，适合本地演示与讲解

## 目录结构

- [frontend](./frontend)
React + Vite + Electron 桌面端

- [backend](./backend)
FastAPI 后端，负责数据处理、训练、统计分析与报告生成

- [demo_datasets](./demo_datasets)
演示与测试数据集

- [docs](./docs)
项目说明、公众号稿、发布文档

- [scripts](./scripts)
打包与清理脚本

## 本地启动

### 1. 安装根目录依赖

```powershell
cd C:\Users\ka\Desktop\ML-app
npm install
```

### 2. 安装前端依赖

```powershell
cd C:\Users\ka\Desktop\ML-app\frontend
npm install
```

### 3. 创建并安装后端虚拟环境

```powershell
cd C:\Users\ka\Desktop\ML-app\backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 4. 从项目根目录启动

```powershell
cd C:\Users\ka\Desktop\ML-app
npm run dev
```

正常情况下不需要单独手动启动后端。

`npm run dev` 会先启动前端和 Electron，Electron 再自动拉起后端 Python 服务。

如果你只想单独调试后端，可以手动运行：

```powershell
cd C:\Users\ka\Desktop\ML-app\backend
.\venv\Scripts\activate
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8010
```

## 打包

### Windows 安装包

```powershell
cd C:\Users\ka\Desktop\ML-app
npm run package:win
```

如果遇到 `EPERM` 或 `Access is denied`，建议使用管理员 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package-win.ps1
```

### 便携版 zip

```powershell
cd C:\Users\ka\Desktop\ML-app
npm run package:portable
```


## 演示数据

演示数据位于：

- [demo_datasets](./demo_datasets)

推荐先看：

- [dataset_catalog.csv](./demo_datasets/dataset_catalog.csv)
- [README.md](./demo_datasets/README.md)
