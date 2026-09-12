# 数据说明（AMPK 数据集与筛选库）

## 文件清单

| 文件 | 说明 |
|------|------|
| `ampk_activators/positive.xlsx` | 阳性集：AMPK 激活剂（含 Smiles 列），共约 298 条原始记录 |
| `ampk_activators/chembl_activators_raw.csv` | 阳性集的 ChEMBL 原始下载文件（含 Assay ChEMBL ID、Assay Description、Assay Organism、Standard Type/Value/Units、pChEMBL Value、Target 等完整元数据） |
| `ampk_inhibitors/negative.xlsx` | 阴性集（含 Smiles 列），共约 250 条原始记录 |
| `ampk_inhibitors/chembl_inhibitors_raw.csv` | 阴性集的 ChEMBL 原始下载文件（字段同上） |
| `all_data.xlsx` | 训练用特征矩阵：Smiles + label + 分子描述符 + 256 位 Morgan 指纹（由根目录 `data_pre_process.py --mode train` 生成） |
| `natural_product_library.xlsx` | 虚拟筛选的天然产物库（食物-药物同源植物来源） |
| `xgb_split.csv` / `rf_split.csv` | 训练/测试划分（由 `save_splits.py` 或训练脚本生成） |

## 数据来源与定义

- **数据库**：ChEMBL（版本：**TODO-待补**），下载原始文件见上表 CSV
- **靶点/物种/亚型**：逐条记录见 CSV 中 `Target` / `Assay Organism` 列（主要为 Homo sapiens，AMP-activated protein kinase）
- **阳性类定义**：对 AMPK 有激活活性（活性阈值标准：**TODO-待补**，如 EC50 < 10 μM）
- **阴性类定义**：**TODO-待补** —— 需作者明确：阴性集为 AMPK 抑制剂、无活性化合物，
  或混合来源；逐条 assay 信息以 `chembl_inhibitors_raw.csv` 为准
- **重复与冲突处理**：原始 ChEMBL 记录可能存在多行同化合物；
  特征矩阵生成时每个 SMILES 保留一条（处理规则：**TODO-待补**）。
  训练/测试集之间的重复与类似物审计见 `results/*_duplicate_audit.csv`

## 特征说明（与论文 Methods 2.2 一致）

- **Morgan 指纹**：ECFP4，radius 2，**256 bits**（列名 `morgan_bit_0` ~ `morgan_bit_255`）
- **分子描述符**：`mol_weight`、`logp`、`tpsa`、`num_h_donors`、`num_h_acceptors`、
  `num_rotatable_bonds`、`num_rings`、`num_aromatic_rings`、`formal_charge`、
  `fraction_csp3`、`num_heteroatoms`、`num_heavy_atoms`、`num_valence_electrons`

## 标签

`label = 1`：阳性（AMPK 激活剂）；`label = 0`：阴性

---

> **发布前必须完成**：将上述 4 处 **TODO-待补** 填写完整
> （ChEMBL 版本、活性阈值、阴性类定义、重复处理规则），
> 这些信息同时需写入论文 Methods 2.2，以回应审稿意见第 2 条。
