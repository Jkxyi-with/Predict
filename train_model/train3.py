# generate_user_risk_scores.py

import pandas as pd
import joblib
from lifelines import CoxPHFitter

def generate_risk_scores(model_path, feature_data_path, prob_data_path, feature_cols, output_path):
    print("加载模型...")
    cph = joblib.load(model_path)

    print("加载特征数据...")
    test_df = pd.read_csv(feature_data_path)

    print("计算 risk_score...")
    risk_scores = cph.predict_partial_hazard(test_df[feature_cols])
    test_df["risk_score"] = risk_scores

    print("加载预测概率（predicted_prob）...")
    prob_df = pd.read_csv(prob_data_path)  # 包含 user_id, predicted_prob

    print("合并风险分与概率...")
    merged_df = pd.merge(prob_df, test_df[["user_id", "risk_score"]], on="user_id", how="left")

    print("按风险分排序...")
    merged_df = merged_df.sort_values("risk_score", ascending=False)

    print("保存至:", output_path)
    merged_df.to_csv(output_path, index=False)

    print("完成！前几条：")
    print(merged_df[["user_id", "predicted_prob", "risk_score"]].head(10))


if __name__ == "__main__":
    # 路径设定
    model_path = "model/cox_model_f.pkl"
    feature_data_path = "/root/lanyun-tmp/prediction/data/test_f.csv"  # 你的测试集特征数据
    prob_data_path = "/root/lanyun-tmp/prediction/accracy_data/features/cox_prediction_test_f.csv"  # 预测结果
    output_path = "coxf_test_user_risk_scores.csv"

    # 指定特征列
    feature_cols = [
    "buy_x", "cart_x", "fav_x", "pv_x",
    "buy_y", "cart_y", "fav_y", "pv_y",
    "total_act",
    "ratio_buy", "ratio_cart", "ratio_fav", "ratio_pv",
    "buy_last1d", "cart_last1d", "fav_last1d", "pv_last1d",
    "buy_last3d", "cart_last3d", "fav_last3d", "pv_last3d"
]


    generate_risk_scores(model_path, feature_data_path, prob_data_path, feature_cols, output_path)
