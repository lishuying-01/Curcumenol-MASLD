import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 1. 参数
# =========================
MODEL_PATH = "model/xgb_best_model.joblib"
DATA_PATH = r"data/predict_data.xlsx"
save_path = "results/xgb_predict_result.xlsx"
TOP_N_FEATURES = 30
SAMPLE_ID = 1
OUT_DIR = "shap_plots"

# 创建输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 2. 加载模型
# =========================
print("加载 XGBoost 模型...")
model = joblib.load(MODEL_PATH)
print("模型加载完成")

# =========================
# 3. 加载预测数据
# =========================
df = pd.read_excel(DATA_PATH)

# 这里只取前两条用于测试
# df = df.iloc[:50].copy()

# 如果带 label，自动去掉
if "label" in df.columns:
    y_true = df["label"]
    df = df.drop(columns=["label"])
else:
    y_true = None

# 保存 Smiles
smiles = df["Smiles"]
X = df.drop(columns=["Smiles"])

print("预测样本数:", X.shape[0])
print("特征维度:", X.shape[1])

# =========================
# 4. 模型预测
# =========================
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

pred_df = pd.DataFrame({
    "Smiles": smiles,
    "pred_label": y_pred,
    "pred_proba": y_proba
})

print("\n预测结果:")
print(pred_df)
print(pred_df.shape)
pred_df.to_excel(save_path, index=False)
print("\n预测结果已保存: xgb_prediction_results.xlsx")

# =========================
# 5. SHAP 解释（稳定推荐写法）
# =========================
print("\n计算 SHAP 值（XGBoost + predict_proba）...")
# explainer = shap.TreeExplainer(model, model_output="probability")
explainer = shap.Explainer(
    model.predict_proba,
    X
)

shap_values = explainer(X)

# =========================
# 关键：二分类只取正类
# shap_values.values: (N, F, 2)
# =========================
shap_vals_pos = shap_values.values[:, :, 1]

print("SHAP shape:", shap_vals_pos.shape)

# =========================
# 6. SHAP 全局解释
# =========================
print("绘制 SHAP summary plot...")

plt.figure()
shap.summary_plot(
    shap_vals_pos,
    X,
    max_display=TOP_N_FEATURES,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "shap_summary_beeswarm.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

print("绘制 SHAP bar plot...")

plt.figure()
shap.summary_plot(
    shap_vals_pos,
    X,
    plot_type="bar",
    max_display=TOP_N_FEATURES,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "shap_summary_bar.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# =========================
# 7. Morgan bit SHAP 重要性
# =========================
mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

morgan_shap = shap_df[
    shap_df["feature"].str.startswith("morgan_bit")
]

print(f"\nTop {TOP_N_FEATURES} Morgan bits (SHAP):")
print(morgan_shap.head(TOP_N_FEATURES))

morgan_shap.head(TOP_N_FEATURES).to_excel(
    "top_morgan_bits_by_shap_xgb.xlsx",
    index=False
)
print("Morgan bit SHAP 排名已保存: top_morgan_bits_by_shap_xgb.xlsx")

# =========================
# 8. 单样本 SHAP 解释
# =========================

print(f"\n绘制单样本 SHAP force plot (sample {SAMPLE_ID})")

# PermutationExplainer 没有 expected_value，手动取 base value
base_value = model.predict_proba(X)[:, 1].mean()

shap.initjs()

plt.figure(figsize=(12, 2))
shap.force_plot(
    base_value,
    shap_vals_pos[SAMPLE_ID],
    X.iloc[SAMPLE_ID],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, f"shap_force_sample_{SAMPLE_ID}.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

print("\n所有 SHAP 图片已保存至:", OUT_DIR)



