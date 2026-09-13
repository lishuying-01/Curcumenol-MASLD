"""
随机森林分类模型训练脚本（与论文建模所用流程完全一致，类结构版本）
====================================================================

流程：
    1. 读取特征矩阵 data/all_data.xlsx
    2. 按 7:3 分层切分训练集与保留测试集（random_state=42）
    3. Pipeline（StandardScaler + 随机森林）5 折网格搜索选参（ROC-AUC）
    4. 保留测试集评估（Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC）
    5. 绘制并保存 ROC / PR 曲线（对应论文 Figure 1A / 1B）
    6. 保存模型至 model/best_random_forest_model.joblib

用法：
    python model_random_forest.py
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换到脚本所在目录（这样相对路径就相对于脚本位置了）
os.chdir(script_dir)

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换到脚本所在目录（这样相对路径就相对于脚本位置了）
os.chdir(script_dir)

DATA_PATH = "../../data/all_data.xlsx"
RESULTS_DIR = "../../results"
MODEL_PATH = "../../model/best_random_forest_model.joblib"
RANDOM_STATE = 42
TRAIN_SIZE = 0.7

# 网格搜索参数空间
PARAM_GRID = {
    "rf__n_estimators": [100, 200, 300],
    "rf__min_samples_leaf": [1,2,4],
    "rf__max_features": ["sqrt", 0.2],
}


class RandomForestPipeline:
    """随机森林二分类模型训练流水线。"""

    def __init__(self, data_path=DATA_PATH, results_dir=RESULTS_DIR,
                 model_path=MODEL_PATH, param_grid=PARAM_GRID,
                 train_size=TRAIN_SIZE, random_state=RANDOM_STATE):
        self.data_path = data_path
        self.results_dir = results_dir
        self.model_path = model_path
        self.param_grid = param_grid
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
        print(f"数据加载完成: {self.data_path}")
        print("特征维度:", self.X.shape)
        print("正负样本比例:")
        print(self.y.value_counts())
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
        print(f"数据切分完成: 训练集={self.X_train.shape}, 测试集={self.X_test.shape}")
        return self

    # =========================
    # 3. 构建 Pipeline（StandardScaler + 随机森林）
    # =========================
    def build_pipeline(self):
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])
        return self

    # =========================
    # 4. 参数搜索（网格搜索 + 5 折交叉验证，ROC-AUC 选参）
    # =========================
    def search_params(self):
        grid = GridSearchCV(
            estimator=self.pipe,
            param_grid=self.param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=2,
        )

        print("开始模型训练 + 参数搜索...")
        grid.fit(self.X_train, self.y_train)

        self.model = grid.best_estimator_

        print("\n最优参数:")
        print(grid.best_params_)
        print("交叉验证最优 ROC-AUC:", grid.best_score_)
        return self

    # =========================
    # 5. 测试集评估
    # =========================
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        print("\n===== 测试集评估结果 =====")
        print(f"Accuracy      : {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Precision     : {precision_score(self.y_test, y_pred):.4f}")
        print(f"Recall        : {recall_score(self.y_test, y_pred):.4f}")
        print(f"F1-score      : {f1_score(self.y_test, y_pred):.4f}")
        print(f"ROC-AUC       : {roc_auc_score(self.y_test, y_proba):.4f}")
        print(f"PR-AUC        : {average_precision_score(self.y_test, y_proba):.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        return self

    # =========================
    # 6. ROC 曲线绘制与保存（论文 Figure 1A）
    # =========================
    def plot_roc_curve(self):
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Random Forest")
        plt.legend(loc="lower right")
        plt.grid(True)

        roc_path = os.path.join(self.results_dir, "roc_curve_random_forest.png")
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nROC 曲线已保存至: {roc_path}")
        return self

    # =========================
    # 7. PR 曲线绘制与保存（论文 Figure 1B）
    # =========================
    def plot_pr_curve(self):
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        precision_arr, recall_arr, _ = precision_recall_curve(self.y_test, y_proba)
        pr_auc = average_precision_score(self.y_test, y_proba)
        positive_rate = self.y_test.mean()

        plt.figure(figsize=(6, 6))
        plt.plot(recall_arr, precision_arr, label=f"PR curve (AP = {pr_auc:.3f})")
        plt.hlines(
            y=positive_rate, xmin=0, xmax=1, linestyles="--",
            label=f"Random baseline (Pos rate = {positive_rate:.3f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve - Random Forest")
        plt.legend(loc="lower left")
        plt.grid(True)

        pr_path = os.path.join(self.results_dir, "pr_curve_random_forest.png")
        plt.savefig(pr_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"PR 曲线已保存至: {pr_path}")
        return self

    # =========================
    # 8. 特征重要性（Top 20）
    # =========================
    def analyze_feature_importance(self, top_n=20):
        rf_model = self.model.named_steps["rf"]
        feature_importance = pd.DataFrame({
            "feature": self.X.columns,
            "importance": rf_model.feature_importances_,
        }).sort_values(by="importance", ascending=False)

        print(f"\nTop {top_n} 特征重要性:")
        print(feature_importance.head(top_n))
        return self

    # =========================
    # 9. 保存模型
    # =========================
    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"\n模型已保存至: {self.model_path}")
        return self

    # =========================
    # 完整流水线
    # =========================
    def run(self):
        return (
            self.load_data()
                .split_data()
                .build_pipeline()
                .search_params()
                .evaluate()
                .plot_roc_curve()
                .plot_pr_curve()
                .analyze_feature_importance()
                .save_model()
        )


if __name__ == "__main__":
    RandomForestPipeline().run()
