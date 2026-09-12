"""
XGBoost 分类模型训练脚本（与论文建模所用流程完全一致，类结构版本）
====================================================================

流程：
    1. 读取特征矩阵 data/all_data.xlsx
    2. 按 7:3 分层切分训练集与保留测试集（random_state=42）
    3. 训练集上做 5 折交叉验证（ROC-AUC，评估稳定性）
    4. 用固定超参数训练最终模型（见下方 XGB_PARAMS，即已发布模型的最终参数）
    5. 保留测试集评估（Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC）
    6. 绘制并保存 ROC / PR 曲线（对应论文 Figure 1C / 1D）
    7. 分析 Morgan 指纹位重要性
    8. 保存模型至 model/xgb_best_model.joblib

用法：
    python xgboost_model.py
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
import joblib

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换到脚本所在目录（这样相对路径就相对于脚本位置了）
os.chdir(script_dir)

DATA_PATH = "../../data/all_data.xlsx"
RESULTS_DIR = "../../results"
MODEL_PATH = "../../model/xgb_best_model.joblib"
RANDOM_STATE = 42
TRAIN_SIZE = 0.7

# 固定超参数 = 已发布模型的最终参数
XGB_PARAMS = dict(
    n_estimators=3,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


class XGBoostPipeline:
    """XGBoost 二分类模型训练流水线。"""

    def __init__(self, data_path=DATA_PATH, results_dir=RESULTS_DIR,
                 model_path=MODEL_PATH, params=XGB_PARAMS,
                 train_size=TRAIN_SIZE, random_state=RANDOM_STATE):
        self.data_path = data_path
        self.results_dir = results_dir
        self.model_path = model_path
        self.params = params
        self.train_size = train_size
        self.random_state = random_state

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        os.makedirs(self.results_dir, exist_ok=True)

    # =========================
    # 1. 读取数据
    # =========================
    def load_data(self):
        df = pd.read_excel(self.data_path)
        self.X = df.drop(columns=["Smiles", "label"])
        self.y = df["label"]
        print(f"数据加载完成: {self.data_path}  样本数={len(self.y)}, 特征数={self.X.shape[1]}")
        return self

    # =========================
    # 2. 数据切分（训练:测试 = 7:3，按类别分层）
    # =========================
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=self.y,
        )
        print(f"数据切分完成: 训练集={len(self.y_train)}, 测试集={len(self.y_test)}")
        return self

    # =========================
    # 3. 构建模型（固定超参数）
    # =========================
    def build_model(self):
        self.model = XGBClassifier(**self.params)
        return self

    # =========================
    # 4. 5 折交叉验证（仅训练集，评估稳定性）
    # =========================
    def cross_validate(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(
            self.model, self.X_train, self.y_train,
            scoring="roc_auc", cv=cv, n_jobs=-1,
        )
        print("\n===== 5-Fold CV ROC-AUC =====")
        print("每折:", scores)
        print("均值:", scores.mean())
        print("标准差:", scores.std())
        return self

    # =========================
    # 5. 训练最终模型
    # =========================
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("\n模型训练完成")
        return self

    # =========================
    # 6. 测试集评估
    # =========================
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        print("\n===== 测试集评估结果 =====")
        print("Accuracy :", accuracy_score(self.y_test, y_pred))
        print("Precision:", precision_score(self.y_test, y_pred))
        print("Recall   :", recall_score(self.y_test, y_pred))
        print("F1-score :", f1_score(self.y_test, y_pred))
        print("ROC-AUC  :", roc_auc_score(self.y_test, y_proba))
        print("PR-AUC   :", average_precision_score(self.y_test, y_proba))

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        return self

    # =========================
    # 7. ROC 曲线绘制与保存（论文 Figure 1C）
    # =========================
    def plot_roc_curve(self):
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - XGBoost")
        plt.legend(loc="lower right")
        plt.grid(True)

        roc_path = os.path.join(self.results_dir, "xgb_roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nROC 曲线已保存: {roc_path}")
        return self

    # =========================
    # 8. PR 曲线绘制与保存（论文 Figure 1D）
    # =========================
    def plot_pr_curve(self):
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        pr_auc = average_precision_score(self.y_test, y_proba)
        baseline = self.y_test.mean()

        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f"PR (AP = {pr_auc:.3f})")
        plt.hlines(
            baseline, 0, 1, linestyles="--",
            label=f"Random baseline (Pos rate = {baseline:.3f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve - XGBoost")
        plt.legend(loc="lower left")
        plt.grid(True)

        pr_path = os.path.join(self.results_dir, "xgb_pr_curve.png")
        plt.savefig(pr_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"PR 曲线已保存: {pr_path}")
        return self

    # =========================
    # 9. Morgan bit 重要性分析
    # =========================
    def analyze_feature_importance(self, top_n=30):
        fi = pd.DataFrame({
            "feature": self.X.columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        morgan_fi = fi[fi["feature"].str.startswith("morgan_bit")]
        print(f"\n===== Top {top_n} Morgan Bits =====")
        print(morgan_fi.head(top_n))
        return self

    # =========================
    # 10. 保存模型
    # =========================
    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"\n模型已保存: {self.model_path}")
        return self

    # =========================
    # 完整流水线
    # =========================
    def run(self):
        return (
            self.load_data()
                .split_data()
                .build_model()
                .cross_validate()
                .train()
                .evaluate()
                .plot_roc_curve()
                .plot_pr_curve()
                .analyze_feature_importance()
                .save_model()
        )


if __name__ == "__main__":
    XGBoostPipeline().run()
