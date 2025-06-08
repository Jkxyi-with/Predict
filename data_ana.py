# analyze_true_event1_predictions.py

import pandas as pd

def analyze_event1_predictions(file_path, threshold=0.05, dataset_name="Test_Day1"):
    """
    分析所有真实为 event=1 的样本，输出它们被模型预测的概率和结果。
    :param file_path: str, CSV路径，应包含 'user_id', 'true_event', 'predicted_prob'
    :param threshold: float, 概率阈值
    :param dataset_name: str, 用于保存输出
    """
    print(f"📥 读取文件：{file_path}")
    df = pd.read_csv(file_path)

    # 添加预测列
    df["predicted_event"] = (df["predicted_prob"] >= threshold).astype(int)

    # 仅保留 true_event == 1 的样本
    positives = df[df["true_event"] == 1].copy()

    # 标记是否预测正确
    positives["correct_prediction"] = (positives["predicted_event"] == 1)

    # 打印前几条记录
    print(f"\n🔍 在 {dataset_name} 中真实 event=1 的用户数量: {len(positives)}")
    print(positives[["user_id", "true_event", "predicted_prob", "predicted_event", "correct_prediction"]].head(10))

    # 保存分析结果
    output_path = f"f_rf_{dataset_name.lower()}_event1_analysis.csv"
    positives.to_csv(output_path, index=False)
    print(f"✅ 保存结果至: {output_path}")


if __name__ == "__main__":
    file_path = "/root/lanyun-tmp/prediction/accracy_data/features/rf_prediction_test_f.csv"
    analyze_event1_predictions(file_path, threshold=0.05, dataset_name="Test_Day1")
