import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# 1. 参数
# =========================
MODEL_PATH = "model/best_random_forest_model.joblib"
DATA_PATH = r"data/predict_data.xlsx"
save_path = "results/rf_predict_result.xlsx"
TOP_N_FEATURES = 30

# =========================
# 2. 加载模型
# =========================
print("加载模型...")
model = joblib.load(MODEL_PATH)

scaler = model.named_steps["scaler"]
rf_model = model.named_steps["rf"]

print("模型加载完成")

# =========================
# 3. 加载数据
# =========================
df = pd.read_excel(DATA_PATH)

if "label" in df.columns:
    df = df.drop(columns=["label"])

smiles = df["Smiles"]
X = df.drop(columns=["Smiles"])

print("预测样本数:", X.shape[0])

# =========================
# 4. 预测
# =========================
X_scaled = scaler.transform(X)

y_pred = rf_model.predict(X_scaled)
y_proba = rf_model.predict_proba(X_scaled)[:, 1]

pred_df = pd.DataFrame({
    "Smiles": smiles,
    "pred_label": y_pred,
    "pred_proba": y_proba
})
pred_df.to_excel(save_path, index=False)
print("预测完成")

