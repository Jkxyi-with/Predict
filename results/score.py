import pandas as pd

def evaluate_ranking(df, k_list=[0.01, 0.05, 0.1], output_path="cox_f_hit.csv"):
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    total = len(df)
    total_event_1 = df["true_event"].sum()
    print(f"全部样本数: {total}, 实际流失用户数: {total_event_1}")

    results = []

    for k in k_list:
        top_n = int(total * k)
        top_df = df.head(top_n)
        hit = top_df["true_event"].sum()
        precision_at_k = hit / top_n
        recall_at_k = hit / total_event_1

        print(f"\nTop {int(k*100)}% ({top_n} 人):")
        print(f"流失命中数: {hit}")
        print(f"Precision@{int(k*100)}%: {precision_at_k:.4f}")
        print(f"Recall@{int(k*100)}%:    {recall_at_k:.4f}")

        results.append({
            "Top %": f"{int(k*100)}%",
            "Top N": top_n,
            "Hit (event=1)": hit,
            "Precision@K": precision_at_k,
            "Recall@K": recall_at_k
        })

    # 保存结果
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")


# 运行
df = pd.read_csv("coxf_test_user_risk_scores.csv")
evaluate_ranking(df, k_list=[0.01, 0.03, 0.05, 0.1], output_path="cox_f_hit.csv")

