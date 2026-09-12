# XGBoost 分子活性分类模型

基于分子描述符 + Morgan 指纹（256 位，ECFP4）特征，使用 XGBoost
（固定超参数）训练二分类模型。
脚本 `xgboost_model.py` 与论文建模所用流程完全一致。

## 训练流程

```
读取特征矩阵 (data/all_data.xlsx)
        │
        ▼
3:7 分层切分（random_state=42）──► 保留测试集（held-out test set，仅最终评估用）
        │
        ▼
训练集 ──► 5 折交叉验证（ROC-AUC，评估稳定性）
        │
        ▼
训练集 ──► 训练最终 XGBoost 模型（固定超参数，见脚本内参数）
        │
        ▼
保留测试集评估（Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC）
        │
        ▼
输出：ROC / PR 曲线（results/）、Top 30 Morgan bits 重要性、模型（model/）
```

## 最终超参数（final selected parameters）

已发布模型（`model/xgb_best_model.joblib`）使用的参数（见脚本）：

| 参数 | 值 |
|------|-----|
| `n_estimators` | 3 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

随机种子固定（42），重新运行脚本将确定性地复现同一模型。

## 使用方法

```bash
cd 开源/train/xgb
python xgboost_model.py
```

环境依赖见包根目录 `requirements.txt`。

## 输出结果

| 文件 | 说明 |
|------|------|
| `results/xgb_roc_curve.png` | ROC 曲线（对应论文 Figure 1C） |
| `results/xgb_pr_curve.png` | PR 曲线（对应论文 Figure 1D） |
| `model/xgb_best_model.joblib` | 已训练模型（虚拟筛选所用） |

带 bootstrap 95% 置信区间与 Cohen's Kappa 的完整评估，请运行包根目录的
`evaluate_with_ci.py`（直接加载已训练模型，无需重新训练）。

上游特征工程见包根目录 `data_pre_process.py`（RDKit 计算描述符与
256 位 Morgan 指纹，与论文 Methods 2.2 一致）。
