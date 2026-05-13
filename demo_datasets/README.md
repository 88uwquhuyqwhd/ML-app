# 演示数据集库

这批数据集用于测试演示，覆盖分类、回归、统计分析、聚类/PCA、脏数据预处理等场景。

## 目录结构

- `classification/` 分类建模示例
- `regression/` 回归建模示例
- `statistics/` 统计分析与聚类/PCA 示例
- `dirty_data/` 缺失值、异常值、日期字段、混合类型示例
- `quick_demo/` 小样本快速演示数据

## 数据集清单

| 文件 | 模块 | 场景 | 行数 | 列数 | 推荐用途 |
| --- | --- | --- | ---: | ---: | --- |
| `classification/subscription_upgrade_cn.csv` | 分类 | SaaS 套餐升级建议 | 1800 | 10 | 中文字段 / 多分类 / 业务演示 |
| `classification/medical_screening_classification.csv` | 分类 | 体检筛查阳性预测 | 1600 | 10 | 分类建模 / 指标可视化 |
| `classification/insurance_claim_fraud_imbalanced.csv` | 分类 | 保险欺诈识别 | 2200 | 12 | 不平衡分类 / 精确率召回率 / 风险识别 |
| `classification/credit_risk_classification.csv` | 分类 | 信贷违约风险 | 2200 | 10 | 二分类 / 逻辑回归 / 随机森林 |
| `classification/hr_attrition_classification.csv` | 分类 | 员工离职预测 | 1500 | 10 | 分类建模 / 特征解释 |
| `classification/student_dropout_classification.csv` | 分类 | 学生流失预测 | 1500 | 10 | 分类建模 / 统计分析 |
| `classification/customer_churn_classification.csv` | 分类 | 电信客户流失 | 1800 | 12 | 分类建模 / 编码 / 缺失值 / 混淆矩阵 |
| `classification/marketing_response_classification.csv` | 分类 | 营销活动响应 | 2100 | 10 | 分类建模 / 分组分析 |
| `classification/equipment_fault_multiclass.csv` | 分类 | 设备健康状态多分类 | 1900 | 9 | 多分类 / 工业预警 / 混淆矩阵 |
| `regression/manufacturing_quality_regression.csv` | 回归 | 制造质量评分 | 1600 | 9 | 回归建模 / 工业场景 |
| `regression/medical_cost_regression.csv` | 回归 | 年度医疗费用预测 | 2000 | 10 | 回归建模 / 医疗业务演示 |
| `regression/housing_price_regression.csv` | 回归 | 房价预测 | 1800 | 11 | 回归建模 / 异常值处理 |
| `regression/logistics_delivery_time_regression.csv` | 回归 | 物流时效预测 | 1700 | 9 | 回归建模 / 业务演示 |
| `regression/ecommerce_order_value_regression.csv` | 回归 | 电商订单金额预测 | 2100 | 10 | 回归建模 / 分类+数值混合 |
| `regression/energy_consumption_regression.csv` | 回归 | 能耗预测 | 1900 | 9 | 回归建模 / 多类型字段 |
| `regression/sales_forecast_regression.csv` | 回归 | 销量预测 | 1700 | 9 | 回归建模 / 时间字段演示 |
| `regression/store_revenue_regression_cn.csv` | 回归 | 门店营业额预测 | 1600 | 10 | 中文字段 / 回归建模 / 线下零售 |
| `quick_demo/quick_classification_demo.csv` | 快速演示 | 快速分类演示 | 180 | 12 | 小样本 / 上手演示 / UI 联调 |
| `quick_demo/quick_regression_demo.csv` | 快速演示 | 快速回归演示 | 200 | 11 | 小样本 / 回归联调 / 快速出图 |
| `quick_demo/quick_statistics_demo.xlsx` | 快速演示 | 快速统计演示 | 160 | 9 | 小样本 / 描述统计 / Excel 上传 |
| `quick_demo/quick_dirty_demo.csv` | 快速演示 | 快速脏数据演示 | 180 | 10 | 小样本 / 清洗联调 / 滚动预览 |
| `statistics/ab_test_experiment.csv` | 统计 | A/B 实验分析 | 2400 | 9 | T 检验 / 方差分析 / 描述统计 |
| `statistics/pca_sensor_monitoring.csv` | 统计 | 传感器主成分分析 | 2000 | 10 | PCA / 相关分析 |
| `statistics/employee_satisfaction_factor_analysis.xlsx` | 统计 | 员工满意度因子分析 | 1000 | 13 | 因子分析 / 描述统计 / Excel 上传 |
| `statistics/training_effect_t_test.csv` | 统计 | 培训效果 T 检验 | 1600 | 7 | 独立样本 T 检验 / 前后差值分析 |
| `statistics/exam_score_correlation_cn.xlsx` | 统计 | 学生成绩相关分析 | 1200 | 9 | 中文字段 / 相关分析 / PCA |
| `statistics/market_research_survey.csv` | 统计 | 市场调研问卷 | 1600 | 11 | 卡方 / 描述统计 / 相关分析 |
| `statistics/ad_spend_linear_regression.csv` | 统计 | 广告投放线性回归 | 1800 | 8 | 线性回归 / 相关分析 / 回归系数解释 |
| `statistics/chi_square_store_preference.csv` | 统计 | 渠道偏好卡方检验 | 1800 | 6 | 卡方 / 交叉表 |
| `statistics/drug_response_anova.csv` | 统计 | 药物组间差异分析 | 1500 | 7 | 方差分析 / 多组比较 / 医疗研究 |
| `statistics/loan_approval_logit.csv` | 统计 | 贷款审批 Logit 回归 | 2000 | 9 | Logit 回归 / 二分类解释 |
| `statistics/retail_segmentation_clustering.csv` | 统计 | 零售客群聚类 | 2400 | 8 | 聚类 / PCA / 描述统计 |
| `dirty_data/dirty_supplier_quality.xlsx` | 脏数据 | 供应商质检脏数据 | 1295 | 10 | 重复值 / 缺失值 / 百分比文本 / Excel 演示 |
| `dirty_data/dirty_customer_profile.csv` | 脏数据 | 客户画像脏数据 | 1500 | 10 | 缺失值 / 异常值 / 日期字段 / 编码 |
| `dirty_data/dirty_property_listing.xlsx` | 脏数据 | 房源信息脏数据 | 1200 | 10 | Excel 上传 / 缺失值 / 异常值 |
| `dirty_data/dirty_order_master.csv` | 脏数据 | 订单主数据脏数据 | 1405 | 10 | 重复值 / 混合编码 / 日期清洗 / 类型识别 |
| `dirty_data/dirty_sales_pipeline.csv` | 脏数据 | 销售漏斗脏数据 | 1300 | 10 | 缺失值 / 异常值 / 筛选 / 派生变量 |
