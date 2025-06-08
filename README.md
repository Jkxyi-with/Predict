# 用户流失预测项目（Cox + XGBoost 等多模型对比）

本项目基于阿里天池提供的用户行为数据集，通过多种建模方法（如 Cox 回归、XGBoost、随机森林、LDA 等）预测用户在电商平台中的流失行为。我们将用户在过去的行为特征转化为风险评分或概率分数，并比较各模型的表现，探索业务可解释性与预测性能的平衡。

---

## 项目背景与目标

- **预测问题**：判断用户是否即将流失（如连续 4 天无行为）
- **应用场景**：电商平台用户留存分析与精准营销
- **项目目标**：探索不同机器学习/统计模型在用户流失预测中的表现差异，寻找性能和可解释性的最佳平衡

---

## 项目结构说明

```
.
├── compare_models.ipynb        # 模型评估对比分析
├── data_ana.py                 # event=1 样本分析
├── data_process.ipynb         # 行为日志预处理与划分
├── data_pro2.ipynb            # 强化版本特征工程
├── read_data.py               # 快速查看数据结构
├── train_features.ipynb       # Cox 有特征训练流程
├── train_xgboost.ipynb        # XGBoost 训练流程
├── train_cox_xg.ipynb         # Cox + XGBoost 组合建模
├── train_linear.ipynb         # Logistic 回归基线模型
├── train.ipynb                # Cox 无详细特征与随机森林建模训练与比较流程
├── train3.py                  # 风险得分计算与排序
├── top_k_recall_comparison.png # Top-K 排名效果图
├── requirement.txt            # Python 依赖包说明
└── README.md                  # 当前说明文件
```

---

## 环境依赖

安装依赖：

```bash
pip install -r requirement.txt
```

`requirement.txt` 内容：

```
pandas
tqdm
jinja2
sklearn
scipy
```

---

## 使用方法

### 1. 数据预处理

将原始 `UserBehavior.csv` 放入 `data/` 目录下，运行以下文件：

```bash
# Step 1: 数据划分（train/val/test）并构建统计特征
运行：data_process.ipynb 或 data_pro2.ipynb

# Step 2: 特征工程
运行：train_features.ipynb
```

### 2. 模型训练与评估

- **XGBoost**：`train_xgboost.ipynb`
- **Cox 回归 + XGBoost**：`train_cox_xg.ipynb`
- **Logistic 回归**：`train_linear.ipynb`
- **完整比较流程**：`train.ipynb`

### 3. 风险分计算

```bash
python train3.py
```

### 4. event=1 分析

```bash
python data_ana.py
```

---

## 模型对比

| 模型类型              | 可解释性 | 预测能力 | 使用方式                                               |
|-----------------------|----------|----------|--------------------------------------------------------|
| **Logistic 回归**         | ✅        | 中等     | 作为简单基线模型，估计流失概率                            |
| **XGBoost**             | ❌        | 高       | 拟合流失事件的概率，擅长处理非线性与特征交互                 |
| **Cox 回归**            | ✅        | 中等偏高 | 建模生存时间与风险因子，输出用户风险评分                     |
| **Cox + XGBoost**       | ✅        | 高       | 基于 Cox 风险评分排序，再融合 XGBoost 预测概率             |
| **随机森林**            | ❌        | 高       | 基于多棵决策树建模用户流失概率，注重召回效果                 |
| **LDA（线性判别分析）** | ✅        | 中等     | 用于判别分析与降维，对特征分布假设较强                      |

---

## 核心特征说明

- **历史行为累积值**：`buy_x`, `cart_x`, `fav_x`, `pv_x`
- **未来窗口行为计数**：`buy_y`, `cart_y`, `fav_y`, `pv_y`
- **活跃总量**：`total_act`
- **近期行为**：`*_last1d`, `*_last3d`
- **行为比例特征**：如 `ratio_buy`, `ratio_cart`

---

## 评估指标

- **Precision@K** / **Recall@K**
- **排序准确性**：通过 Top-N 用户中命中流失事件的比例评估
- **概率输出（可选）**：如 AUC、log-loss（适用于 XGBoost、Logistic）

---

## 潜在改进方向

- 引入 LSTM 等时间序列模型处理行为序列
- 基于更多，最近且时间跨度的数据集进行模型训练
- 利用注意力机制/图神经网络挖掘行为关系
- 结合用户画像与多源数据如评论情感分析等增强预测能力
- 基于流失原因做 Error Analysis 和可解释性分析

---

## 团队分工

- Jin Xuanyi：代码调试、模型训练
- Wang Kexin：数据预处理、报告撰写

---

## 参考文献

1. D. R. Cox, “Regression Models and Life-Tables,” *Journal of the Royal Statistical Society*, 1972.
2. 阿里天池用户行为数据集：https://tianchi.aliyun.com/dataset/649
3. Lifelines 生存分析工具：https://lifelines.readthedocs.io/
4. 吴下阿泽. (2021, August 16). 利用生存分析 Kaplan‑Meier 法与 COX 比例风险回归模型进行客户流失分析与剩余价值预测. CSDN博客。检索自 https://blog.csdn.net/maiyida123/article/details/119736185
5. Li, M. (2022, November 1). Research on the prediction of e‑commerce platform user churn based on Random Forest model. In *Proceedings of the 3rd International Conference on Computer Science and Management Technology (ICCSMT)* (pp. 34–39). IEEE. https://doi.org/10.1109/ICCSMT58129.2022.00014

