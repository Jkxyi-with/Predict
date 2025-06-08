import pandas as pd
import matplotlib.pyplot as plt

# 加载三个 hit 结果表格
cox_df = pd.read_csv("accuracy_data/hit&rank/cox_f_hit.csv")
rf_df = pd.read_csv("accuracy_data/hit&rank/rf_hit.csv")
xgb_df = pd.read_csv("accuracy_data/hit&rank/xgb_hit.csv")
xgb_cox_df = pd.read_csv("accuracy_data/hit&rank/xgb_with_cox_hit.csv")

# 统一 Top % 作为 x 轴（数值化）
def process(df, name):
    df = df.copy()
    df["Top %"] = df["Top %"].str.replace("%", "").astype(float)
    df["model"] = name
    return df

cox_df = process(cox_df, "Cox")
rf_df = process(rf_df, "Random Forest")
xgb_df = process(xgb_df, "XGBoost")
xgb_cox_df = process(xgb_cox_df, "XGBoost + Cox")

# 合并数据
all_df = pd.concat([cox_df, rf_df, xgb_df, xgb_cox_df], ignore_index=True)

# 绘制 Recall@K 曲线
plt.figure(figsize=(10, 6))
for model_name, group in all_df.groupby("model"):
    plt.plot(group["Top %"], group["Recall@K"], label=model_name, marker="o")

plt.title("Top-K Recall Comparison")
plt.xlabel("Top % of Users")
plt.ylabel("Recall@K")
plt.xticks(group["Top %"])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# 保存图表
plt.savefig("top_k_recall_comparison.png")
# 保存合并后的数据
all_df.to_csv("accuracy_data/hit&rank/combined_hit_results.csv", index=False)
# 输出合并后的数据预览
print("合并后的数据预览：")
print(all_df.head())
# 输出每个模型的 Recall@K 最佳值
print("\n每个模型的 Recall@K 最佳值：")
for model_name, group in all_df.groupby("model"):
    best_recall = group["Recall@K"].max()
    print(f"{model_name}: {best_recall:.4f} at Top {group.loc[group['Recall@K'].idxmax(), 'Top %']}%")  