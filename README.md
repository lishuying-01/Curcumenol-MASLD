# Integrative Machine Learning Screening Identifies Curcumenol as a Novel AMPK Signaling Modulator Alleviating MASLD — Model Code & Data

## 1. 研究概述

| 项目 | 内容 |
|------|------|
| 任务 | 二分类：预测化合物是否为 AMPK 激活剂（阳性，label = 1） |
| 特征 | 13 个分子描述符 + 256 位 Morgan 指纹（ECFP4，radius = 2） |
| 模型 | Random Forest（Pipeline + 网格搜索调参）与 XGBoost（固定超参数） |
| 切分 | 分层随机切分（训练:测试 = 7:3，random_state = 42） |
| 筛选 | 对天然产物库（食物-药物同源植物来源）预测并排名，Curcumenol 排名第一 |

## 2. 目录结构

```
├── README.md                        本文件
├── LICENSE                          MIT 开源协议
├── requirements.txt                 Python 依赖
│
├── data/                            数据（详见 data/README.md）
│   ├── train_data.xlsx              训练特征矩阵（Smiles + label + 13 描述符 + 256 位 Morgan 指纹）
│   ├── raw_predict_data.xlsx        待预测化合物（含 Smiles 列）
│   ├── predict_data.xlsx            待预测化合物的特征矩阵（data_pre_process.py 生成）
│   ├── natural_product_library.xlsx 虚拟筛选的天然产物库
│   ├── ampk_activators/             阳性集 positive.xlsx + ChEMBL 原始下载 chembl_activators_raw.csv
│   ├── ampk_inhibitors/             阴性集 negative.xlsx + ChEMBL 原始下载 chembl_inhibitors_raw.csv
│   └── README.md                    数据来源与特征说明
│
├── data_pre_process.py              特征生成（RDKit：描述符 + 256 位 Morgan 指纹）
├── predict_random_forest.py         用已训练 RF 模型预测（输出概率）
├── predict_xgb.py                   用已训练 XGB 模型预测 + SHAP 可解释性分析
│
├── train/                           训练脚本（确定性复现模型）
│   ├── rf/model_random_forest.py    随机森林训练（网格搜索调参）
│   ├── rf/README.md
│   ├── xgb/xgboost_model.py         XGBoost 训练（固定超参数）
│   └── xgb/README.md
│
├── model/                           已训练模型（可直接加载，无需重新训练）
│   ├── best_random_forest_model.joblib   Pipeline（StandardScaler + RF）
│   └── xgb_best_model.joblib             XGBClassifier
│
├── results/                         运行输出
└── shap_plots/                      SHAP 图表（predict_xgb.py 生成）
```

## 3. 环境安装

```bash
pip install -r requirements.txt
```

依赖：rdkit、scikit-learn、xgboost、shap、pandas、numpy、matplotlib、joblib、openpyxl（建议 Python ≥ 3.9）。

## 4. 使用流程

### 4.1 特征生成（data_pre_process.py）

`calculate_molecular_features()` 对每个 SMILES 依次执行：

1. RDKit 解析 SMILES（失败则跳过）；
2. `SaltRemover` 去盐；
3. 计算 13 个分子描述符（见 §5.1）；
4. 计算 256 位 Morgan 指纹（radius = 2，列名 `morgan_bit_0` ~ `morgan_bit_255`）。

两种模式（在脚本 `__main__` 中切换）：

- **训练数据**：`train_process()` 读取阳性集 `data/ampk_activators/positive.xlsx` 与
  阴性集 `data/ampk_inhibitors/negative.xlsx`，分别计算特征后合并（阳性 label = 1，
  阴性 label = 0），输出训练特征矩阵（默认调用已注释，路径按需修改）；
- **预测数据**：`predict_process()` 读取 `data/raw_predict_data.xlsx`，
  输出 `data/predict_data.xlsx`（label 列为占位 0，仅用于对齐列格式）。

```bash
python data_pre_process.py
```

### 4.2 训练模型（可选，用于复现）

两个训练脚本均：读取训练特征矩阵 → 7:3 分层切分（random_state = 42）→
训练 → 保留测试集评估（Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC、
混淆矩阵、分类报告）→ 绘制并保存 ROC / PR 曲线 → 保存模型至 `model/`。

```bash
cd train/rf  && python model_random_forest.py   # 随机森林
cd train/xgb && python xgboost_model.py         # XGBoost
```

- **Random Forest**：Pipeline（StandardScaler + `class_weight='balanced'` 的 RF），
  训练集上网格搜索 + 5 折交叉验证（ROC-AUC 选参）；
- **XGBoost**：训练前先在训练集上做 5 折交叉验证评估稳定性，
  再用固定超参数训练最终模型。

> **注意**：训练脚本中 `DATA_PATH = "../../data/all_data.xlsx"`，
> 本仓库提供的特征矩阵文件名为 `data/train_data.xlsx`。
> 运行前请将该文件重命名为 `all_data.xlsx`，或修改脚本中的 `DATA_PATH`。

### 4.3 预测与 SHAP 分析

两个预测脚本均直接加载 `model/` 下已训练模型，读取 `data/predict_data.xlsx`，
输出每个化合物的 `pred_label`（预测类别）与 `pred_proba`（预测为阳性的概率）。

```bash
python predict_random_forest.py   # 输出 results/rf_predict_result.xlsx
python predict_xgb.py             # 输出 results/xgb_predict_result.xlsx + SHAP 分析
```

`predict_xgb.py` 额外执行 SHAP 可解释性分析（`shap.Explainer(model.predict_proba, X)`，
二分类取正类 SHAP 值）：

| 输出 | 说明 |
|------|------|
| `shap_plots/shap_summary_beeswarm.png` | SHAP 蜂群图（Top 30 特征） |
| `shap_plots/shap_summary_bar.png` | SHAP 全局特征重要性（Top 30） |
| `shap_plots/shap_force_sample_1.png` | 单样本 force plot |
| `top_morgan_bits_by_shap_xgb.xlsx`（仓库根目录） | Morgan 指纹位按 SHAP 重要性排名（Top 30） |

## 5. 方法学要点

### 5.1 特征（与论文 Methods 2.2 一致）

- **分子描述符（13 个）**：`mol_weight`、`logp`、`tpsa`、`num_h_donors`、
  `num_h_acceptors`、`num_rotatable_bonds`、`num_rings`、`num_aromatic_rings`、
  `formal_charge`、`fraction_csp3`、`num_heteroatoms`、`num_heavy_atoms`、
  `num_valence_electrons`
- **Morgan 指纹**：256 bits，radius = 2（ECFP4），生成代码见 `data_pre_process.py`
  （`nBits=256`）。

### 5.2 数据切分

分层随机切分，固定种子 `random_state = 42`，训练:测试 = 7:3。
此为随机切分产生的内部保留测试集（held-out test set），仅用于最终评估。

### 5.3 超参数

- **Random Forest**：网格搜索空间见 `train/rf/model_random_forest.py` 的 `PARAM_GRID`
  （`n_estimators` 100/200、`max_depth` 2/3/4、`min_samples_split` 10/20、
  `min_samples_leaf` 30/40、`max_features` sqrt/0.2），最优参数在运行时打印。
- **XGBoost**：固定超参数（见 `train/xgb/xgboost_model.py` 的 `XGB_PARAMS`：
  `n_estimators=3`、`max_depth=4`、`learning_rate=0.05`、`subsample=0.8`、
  `colsample_bytree=0.8`），训练前在训练集上做 5 折交叉验证评估稳定性。
- 已发布模型的完整参数导出见 `results/rf_final_params.json` 与
  `results/xgb_final_params.json`。

### 5.4 预处理与类别不平衡

- Random Forest 的 Pipeline 中对特征做 StandardScaler（Z-score）标准化；
  `class_weight='balanced'` 处理类别不平衡。
- XGBoost 直接使用原始特征（树模型对量纲不敏感）。

## 6. 输出文件一览

| 文件 | 生成脚本 | 说明 |
|------|----------|------|
| `data/train_data.xlsx` | `data_pre_process.py` | 训练特征矩阵 |
| `data/predict_data.xlsx` | `data_pre_process.py` | 预测特征矩阵 |
| `model/best_random_forest_model.joblib` | `train/rf/model_random_forest.py` | RF 模型 |
| `model/xgb_best_model.joblib` | `train/xgb/xgboost_model.py` | XGB 模型 |
| `results/roc_curve_random_forest.png`、`pr_curve_random_forest.png` | `train/rf/...` | RF 的 ROC / PR 曲线（论文 Figure 1A / 1B） |
| `results/xgb_roc_curve.png`、`xgb_pr_curve.png` | `train/xgb/...` | XGB 的 ROC / PR 曲线（论文 Figure 1C / 1D） |
| `results/rf_predict_result.xlsx` | `predict_random_forest.py` | RF 预测结果（Smiles + pred_label + pred_proba） |
| `results/xgb_predict_result.xlsx` | `predict_xgb.py` | XGB 预测结果 |
| `shap_plots/*.png`、`top_morgan_bits_by_shap_xgb.xlsx` | `predict_xgb.py` | SHAP 图表与 Morgan 位重要性 |

## 7. 许可

本仓库以 **MIT License** 开源，见 [LICENSE](LICENSE)。

## 8. 引用

如使用本代码或数据，请引用论文：

> Li S., Chen R., Hou K., Zhao R., Li Y., Li S. Integrative machine learning
> screening and experimental validation identify Curcumenol as a novel AMPK
> signaling modulator alleviating MASLD.（论文正式发表信息待补）
