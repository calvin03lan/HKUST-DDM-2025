# Project 2: Bayesian Logistic Regression - 项目执行计划

**截止日期:** 2026年5月22日 23:59  
**技术栈:** Python (纯 `.py` 脚本), LaTeX  
**报告作者:** LAN, Tianwei (ID: 21230969)  
**报告语言:** English

## 阶段一：理论推导与准备 (Theoretical Derivations)
*此阶段不涉及代码，重点是完成数学推导，为后续的代码实现和报告撰写打下基础。*

- **任务 1.1：推导一维拒绝采样常数 (Part 1 a & b)**
  - 证明为了使 $f(w) = A e^{-\lambda|w|}$ 成为有效的比较函数，必须满足 $A \ge \frac{1}{\sqrt{2\pi}} e^{\lambda^2/2}$。
  - 计算理论接受率（目标分布面积 / 提议分布面积），并用 $A$ 和 $\lambda$ 表达。
- **任务 1.2：求解最优比较函数 (Part 1 c)**
  - 找到使接受率最大化的最优 $\lambda$。
  - 代入求出最优的 $f(w)$，即 $f(w) = \sqrt{\frac{e}{2\pi}} e^{-|w|}$，并计算最终的最佳接受率。
- **任务 1.3：将推导过程录入 LaTeX**
  - 在 LaTeX 模板中创建 `Part 1` 章节。
  - 将上述推导的数学公式使用 LaTeX 语法排版，确保逻辑严密、学术规范。

## 阶段二：核心代码编写与测试 (Code Implementation)
*在项目目录下创建三个主要的 Python 脚本：`part1.py`, `part2.py`, `part3.py`。所有绘图结果应直接保存为本地图片，以便稍后插入 LaTeX。*

### 模块 1: 一维拒绝采样 (`part1.py`)
- **任务 2.1.1：编写基础采样函数 (Part 1 d & e)**
  - 编写拒绝采样代码生成标准正态分布 $\mathcal{N}(0, 1)$。
  - 运行代码生成样本，验证实际接受率。
  - 绘制样本直方图，叠加理论标准正态分布曲线并保存为 `part1_standard_normal.png`。
- **任务 2.1.2：推广至一般正态分布 (Part 1 f)**
  - 修改代码，使其支持生成任意 $\mathcal{N}(\mu, \sigma^2)$。

### 模块 2: 数值积分 (`part2.py`)
- **任务 2.2.1：数据生成 (Part 2 a)**
  - 生成 $N=500$ 的数据集：$x_i \sim \mathcal{N}(1, 1)$，真实权重 $w_{true} = [-1, 1]^T$。
  - 添加噪声 $\epsilon_i \sim \mathcal{N}(0, 4)$（标准差 $\sigma = 2$）。
- **任务 2.2.2：一维积分实现 (Part 2 b & c)**
  - 编写 **梯形法则 (Trapezoidal rule)** 代码。
  - 编写 **龙贝格积分 (Romberg integration)** 代码。
- **任务 2.2.3：二维积分实现 (Part 2 d & e)**
  - 编写 **辛普森法则 (Simpson's rule)** 双重积分代码。
  - 编写 **粗糙蒙特卡洛 (Crude MC)** 积分代码。

### 模块 3: MCMC 与 MAP 优化 (`part3.py`)
- **任务 2.3.1：Metropolis 算法 (Part 3 a)**
  - 编写 Metropolis 采样器，调整 $\sigma_p$ 使接受率保持在 ~30%。
  - 绘制采样结果分布图并保存为 `part3_metropolis.png`。
- **任务 2.3.2：模拟退火寻找 MAP (Part 3 b)**
  - 定义能量函数：$E(w) = -\log[p(D \mid w) p(w)]$。
  - 实现模拟退火算法寻找全局最优解 $w_{MAP}$。
- **任务 2.3.3：重要性采样 (Part 3 c)**
  - 构建新的提议分布 $q(w)$ 运行蒙特卡洛积分计算边际似然。

## 阶段三：LaTeX 报告整合与排版 (Report Writing)
- 撰写 Part 1 结果。
- 撰写 Part 2 结果。
- 撰写 Part 3 结果。

## 阶段四：最终审查与交付 (Final Review & Submission)
- 代码审查与清理。
- 报告审校。
- 打包提交。
