MSDM 5055 课程作业 第一部分
2026年春季学期 
**深度学习建模：概念、工具与技术** 
---
第一部分作业说明 

* 前三个问题基于提供的模板 `mnist Classification.py`，分类准确率使用 `accurate Checker1.py` 计算。
* 第 4 题提供模板 `pytorchMnist.py`，分类准确率使用 `accurate Checker2.py` 计算。
* 第 5 题提供模板 `cifar10Classification.py`，分类准确率使用 `accurateChecker3.py` 计算。
* 对于前四个问题，可以在截止日期后一周内提交，但逾期提交只能获得 80% 的分数。
---

1. 自动微分 (Autodiff) 
* (a) 实现 Softmax 函数的前向和反向传播过程：对应 `mnist Classification.py` 的第 105 行和第 111 行 。
$$SoftMax(x_{i})=\frac{\exp x_{i}}{\sum_{k}\exp x_{k}}$$
* (b) 实现交叉熵损失函数 (Cross Entropy Loss) 的前向和反向传播过程**：对应 `mnist Classification.py` 的第 125 行和第 131 行 。
$$CorssEntropy(x_{i},l_{i})=-\sum_{i}l_{i}\log(x_{i})$$
* 截止日期：2026年3月22日 

2. 优化器 (Optimizer) 
* 请实现 **SGD（随机梯度下降）** 优化过程：对应 `mnist Classification.py` 的第 294 行 。
* 截止日期：2026年3月22日 

3. 训练 (Training) 
* (a) 实现线性层类的 Xavier 初始化：对应 `mnist Classification.py` 的第 36 行 。
* (b) 微调超参数：使分类准确率至少达到 **94%** 。
* 只能使用模板中提供的层类型和损失函数 。

* 神经网络的参数总数不得超过 $2 \times 10^{5}$ 。
* 截止日期：2026年3月22日 

4. Pytorch 实战 
* 请实现 **MNIST 分类** 的 Pytorch 版本，并使准确率至少达到 **97%** 。
* 仅限使用全连接线性层 (FC linear layers) 。
* 神经网络的参数总数不得超过 $6 \times 10^{5}$ 。
* 截止日期：2026年4月5日 

5. 竞赛一 (Competition One) 
* 请实现 **CIFAR-10 分类** 的 Pytorch 版本，准确率需至少达到 **60%** 以获得基础分 。
* 评分标准：你的最终得分将根据 $\frac{\text{accuracy} \times 10^2}{\max(\text{number of parameters} \times 10^{-4}, 1)}$ 计算 。
* 所有提交的作品将进行排名，排名越高，得分越多 。
* 截止日期：2026年4月26日 