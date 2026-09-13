# 随机森林分子活性分类模型

基于分子描述符 + Morgan 指纹（256 位，ECFP4）特征，使用随机森林
（Pipeline + 网格搜索调参）训练二分类模型。
脚本 `model_random_forest.py` 与论文建模所用流程完全一致。

## 训练流程

```
读取特征矩阵 (data/all_data.xlsx)
        │
        ▼
4:6 分层切分（random_state=42）──► 保留测试集（held-out test set，仅最终评估用）
        │
        ▼
训练集 ──► Pipeline(StandardScaler + RF) 5 折网格搜索
           （ROC-AUC 选参，最优参数组合作为最终模型）
        │
        ▼
保留测试集评估（Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC）
        │
        ▼
输出：ROC / PR 曲线（results/）、Top 20 特征重要性、模型（model/）
```

## 超参数搜索空间（hyperparameter search space）

网格搜索在训练集上进行（5 折交叉验证，ROC-AUC 选参）：

| 参数 | 候选值 |
|------|--------|
| `n_estimators` | 100, 200,300|
| `min_samples_leaf` | 1,2,4 |
| `max_features` | sqrt, 0.2 |

最优参数（final selected parameters）在运行时打印。

## 使用方法

```bash
cd 开源/train/rf
python model_random_forest.py
```

环境依赖见包根目录 `requirements.txt`。

## 输出结果

| 文件 | 说明 |
|------|------|
| `results/rrandom_forest_roc_curve.png` | ROC 曲线（对应论文 Figure 1A） |
| `model/best_random_forest_model.joblib` | 最优 Pipeline（Scaler + RF） |

