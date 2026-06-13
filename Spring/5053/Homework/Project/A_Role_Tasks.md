# MSDM5053 项目 - A 角色任务清单

## 基本信息

- **负责成员**: A
- **负责股票**: HSBC Holdings (0005.HK)
- **行业**: 银行/金融
- **选择理由**: 港股传统蓝筹；受利率、金融周期、全球风险影响；适合 GARCH 波动建模

---

## 一、数据准备任务

| 任务 | 说明 |
|------|------|
| 下载数据 | 下载 0005.HK (HSBC) 的日度 adjusted close price |
| 时间范围 | 2011-01-03 至 2024-12-31 |
| 数据保存 | 统一保存为 CSV 格式 |
| 变量计算 | 计算 log_price = ln(P_t) 和 log return = 100 * [ln(P_t) - ln(P_{t-1})] |
| 数据集划分 | 训练集：2011-2022，测试集：2023-2024（或使用最后 20%作为测试集）|

---

## 二、独立分析任务（单变量分析）

### 2.1 描述性统计分析
- [ ] 计算描述统计量：mean, sd, min, max, skewness, kurtosis, Jarque-Bera/normality
- [ ] 绘制图 1：Adjusted close price / log price plot（展示长期趋势）
- [ ] 绘制图 2：Log return plot（展示波动聚集、极端收益日）
- [ ] 分析收益率分布特征和极端值

### 2.2 平稳性与自相关检验
- [ ] ADF 检验（验证 log return 的平稳性）
- [ ] 绘制图 3：ACF/PACF of returns（判断序列相关与 ARMA 候选阶数）
- [ ] Ljung-Box 检验（检验收益率序列相关性）

### 2.3 ARMA 均值模型
- [ ] 根据 ACF/PACF 和 AIC/BIC 选择 ARMA(p,q) 模型
- [ ] 检查残差是否近似 white noise
- [ ] 填写表 3：ARMA candidates（候选模型的 AIC/BIC 和残差检验）

### 2.4 波动率模型
- [ ] ARCH effect test（对 ARMA 残差平方做 ARCH-LM 或 Ljung-Box 检验）
- [ ] 绘制图 4：ACF of squared returns（观察 ARCH/GARCH 效应）
- [ ] 估计 GARCH(1,1) 模型（建模条件方差与 volatility clustering）
- [ ] 估计 EGARCH 或 GJR-GARCH（检验负收益是否对波动有更强影响）
- [ ] 绘制图 5：Fitted conditional volatility（展示 GARCH/EGARCH 估计出的波动变化）
- [ ] 填写表 4：GARCH-type models（参数、AIC/BIC、alpha+beta 或 leverage 项）
- [ ] 解释波动持续性和非对称效应

### 2.5 预测分析
- [ ] 使用测试集比较模型表现
- [ ] 比较 ARMA 和 ARMA-GARCH 在测试集上的 RMSE/MAE/方向准确率
- [ ] 绘制图 6：Forecast vs actual / forecast error
- [ ] 填写表 5：Forecasting performance
- [ ] 解释预测是否有实际意义

### 2.6 个股结论
- [ ] 总结 HSBC 的可预测性、波动性和相对特点
- [ ] 重点解释银行股波动与利率/金融风险的关系

---

## 三、必须产出的图表清单

### 图表（6 个）

| 编号 | 图表名称 | 用途 |
|------|----------|------|
| 图 1 | Adjusted close price / log price plot | 展示长期趋势；说明股价水平可能非平稳 |
| 图 2 | Log return plot | 展示波动聚集、极端收益日 |
| 图 3 | ACF/PACF of returns | 判断收益率序列相关与 ARMA 候选阶数 |
| 图 4 | ACF of squared returns | 观察 ARCH/GARCH 效应 |
| 图 5 | Fitted conditional volatility | 展示 GARCH/EGARCH 估计出的波动变化 |
| 图 6 | Forecast vs actual / forecast error | 展示测试期预测表现 |

### 表格（5 个）

| 编号 | 表格名称 | 内容 |
|------|----------|------|
| 表 1 | 描述统计 | mean, sd, min, max, skewness, kurtosis, Jarque-Bera/normality |
| 表 2 | Stationarity and autocorrelation tests | ADF；Ljung-Box on returns；Ljung-Box/ARCH-LM on squared residuals |
| 表 3 | ARMA candidates | 候选 ARMA(p,q)的 AIC/BIC 和残差检验 |
| 表 4 | GARCH-type models | GARCH(1,1)与 EGARCH/GJR 的参数、AIC/BIC、alpha+beta 或 leverage 项 |
| 表 5 | Forecasting performance | RMSE、MAE、directional accuracy；比较 ARMA 和 ARMA-GARCH |

---

## 四、合并分析任务（需协作完成）

| 任务 | 说明 | 依赖 |
|------|------|------|
| **横向比较表** | 负责汇总整理四只股票的描述统计、最佳 ARMA、最佳 GARCH、预测指标 | 个人结果表 |
| 数据提供 | 向 D 提供一份含 Date 和 log return 的 CSV 用于数据合并 | - |

---

## 五、演示分工任务

| 项目 | 内容 | 建议时长 |
|------|------|----------|
| 演示部分 | Introduction + Data + HSBC 重点发现 | 3-4 分钟 |

---

## 六、最终交付物清单

- [ ] HSBC 小节（按模板格式撰写）
- [ ] 图表（图 1-6）
- [ ] 模型结果表（表 1-5）
- [ ] 预测结果
- [ ] 横向比较汇总表（四只股票对比）
- [ ] 保存代码和数据处理过程（用于附录）

---

## 七、报告小节撰写模板

按照以下模板撰写 HSBC 小节：

1. **Data description**: 简要介绍股票行业、样本区间、价格和收益率图
2. **Preliminary statistics**: 描述统计、收益率分布、极端值和波动聚集现象
3. **Stationarity and serial correlation**: ADF、ACF/PACF、Ljung-Box 结果
4. **ARMA mean model**: 根据 ACF/PACF 和 AIC/BIC 选择 ARMA(p,q)，检查残差是否近似 white noise
5. **Volatility model**: 先做 ARCH effect test，再估计 GARCH(1,1)和 EGARCH/GJR，解释波动持续性和非对称效应
6. **Forecasting**: 使用测试集比较模型表现，给出 RMSE/MAE/方向准确率，并解释预测是否有实际意义
7. **Individual conclusion**: 总结该股票的可预测性、波动性和相对特点

---

## 八、核心模型与技术方法

A 角色需要掌握和应用以下课程技术方法：

- ACF/PACF 分析
- ADF 单位根检验
- Ljung-Box 检验
- ARMA/ARIMA 模型
- ARCH effect test (ARCH-LM)
- GARCH(1,1) 模型
- EGARCH/GJR-GARCH 模型（至少二选一）
- 预测评估 (RMSE, MAE, directional accuracy)

---

## 九、注意事项

1. 不要只做 2019-2024 五年样本，确保使用 2011-2024 完整区间
2. 不要只报告模型参数，必须解释金融含义（如波动持续性、负冲击效应）
3. 每人必须保存代码和数据处理过程，附录中应放支持材料
4. 重点关注银行股波动与利率/金融风险的关系
5. SARIMA 和 VECM 可作为可选/附录，主线应是 ARMA + GARCH + Forecasting

---

*文档基于 MSDM5053_final_division_plan.pdf 整理生成*
