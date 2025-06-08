import pandas as pd

# 查看 Cox 模型输出文件的列名
cox_path = "/root/lanyun-tmp/prediction/data/UserBehavior.csv"
cox_df = pd.read_csv(cox_path)

# 输出列名
print(list(cox_df.columns))
