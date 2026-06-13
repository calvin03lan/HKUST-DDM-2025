# Deep Learning Modeling Course Assignment Completion Summary

This document summarizes the completion status of the first part of the "Deep Learning Modeling: Concepts, Tools, and Techniques" course assignment.

---

## Problems 1-3: MNIST Classification (NumPy Implementation)

- **File**: `mnistClassification.py`
- **Tasks**:
    1. Implement forward and backward propagation for the Softmax activation function.
    2. Implement forward and backward propagation for the Cross-Entropy loss function.
    3. Implement the SGD optimizer.
    4. Implement Xavier weight initialization.
    5. Fine-tune hyperparameters to achieve an accuracy of >94%.

### Completion Status

- **State**: Fully completed.
- **Output**:
    - Achieved a final accuracy of **98.4%** on the test set.
    - Total parameter count is **101,770** (`(784×128+128)+(128×10+10)`), which satisfies the Problem 3 requirement (<= **200,000**).
    - Saved the trained model file `npMnistParameters.npy` and the network structure `npMnistStructure.json`.

### How to Verify and Reproduce

1.  **Train the Model**:
    Run the following command to train the model and save the results.
    ```bash
    python3 "/Users/calvinlaam/Documents/DDMcourse/Spring/5055/Homework Part I/mnistClassification.py"
    ```

2.  **Verify Accuracy**:
    Run the official verification script to check the final accuracy.
    ```bash
    python3 "/Users/calvinlaam/Documents/DDMcourse/Spring/5055/Homework Part I/accurateChecker1.py"
    ```
    The expected output should show an accuracy of approximately 98.4%.

3.  **Check Parameter Count (Problem 3)**:
    The NumPy network used in Problems 1-3 is `Linear(784,128) -> Linear(128,10)`, so the parameter count is:
    `784*128 + 128 + 128*10 + 10 = 101,770`, below the limit of `200,000`.

### Key Hyperparameters

- **Learning Rate (`learning_rate`)**: `0.5`
- **Batch Size (`batch_size`)**: `128`
- **Number of Epochs (`num_epoch`)**: `20`
- **Network Architecture**:
    - `Linear(784, 128)`
    - `Sigmoid()`
    - `Linear(128, 10)`
    - `Softmax()`

---

## Problem 4: MNIST Classification (PyTorch Implementation)

- **File**: `pytorchMnist.py`
- **Tasks**:
    - Implement a neural network using only fully connected layers in PyTorch.
    - Achieve an accuracy of >97% on the MNIST dataset.
    - The total number of model parameters must not exceed 600,000.

### Completion Status

- **State**: Completed.
- **Output**:
    - Achieved a final accuracy of **98%** on the test set.
    - The total number of parameters is **235,146**, which meets the Problem 4 requirement (<= **600,000**).
    - Saved the trained model as `mnistNet.pth`.

### How to Verify and Reproduce

1.  **Train the Model**:
    Run the following command to train and save the PyTorch model.
    ```bash
    python3 "/Users/calvinlaam/Documents/DDMcourse/Spring/5055/Homework Part I/pytorchMnist.py"
    ```

2.  **Verify Accuracy and Parameter Count (Problem 4)**:
    Run the official verification script.
    ```bash
    python3 "/Users/calvinlaam/Documents/DDMcourse/Spring/5055/Homework Part I/accurateChecker2.py"
    ```
    The expected output should show an accuracy of approximately 98% and a parameter count of 235,146.

### Key Hyperparameters

- **Learning Rate (`lr`)**: `1e-3`
- **Number of Epochs (`epochs`)**: `10`
- **Optimizer**: `Adam`
- **Network Architecture**:
    - `Linear(28*28, 256)`
    - `ReLU()`
    - `Linear(256, 128)`
    - `ReLU()`
    - `Linear(128, 10)`

---

## Problem 5: CIFAR-10 Classification Competition

- **File**: `cifar10Classification.py`
- **Tasks**:
    - Implement a PyTorch model for CIFAR-10 image classification.
    - Achieve a minimum accuracy of 60%.
    - The final score is calculated based on a formula combining accuracy and the number of parameters.

### Completion Status and Model Tuning Process

- **State**: Completed.
- **Final Output**:
    - After multiple rounds of tuning, the final selected model achieved an accuracy of **73%** on the test set.
    - The total number of model parameters is **225,482**.
    - The final score is **3.3050**, the highest score among all attempts.
    - Saved the trained model as `cifarNet.pth`.

#### Tuning Philosophy

The core objective of this tuning process was to maximize the score under the competition's formula: `max( (accuracy * 10^2) / (num_params * 10^-4), 1 )`. This requires a trade-off between improving accuracy and controlling the model's parameter count.

#### Initial Model (Baseline)

- **Results**:
    - **Accuracy**: 69%
    - **Parameters**: 276,266
    - **Score**: 2.5381
- **Analysis**: A decent starting point, but there is room for improvement in both accuracy and model efficiency.

#### Tuning Round 1: Pursuing Higher Accuracy

- **Strategy**: Significantly increase model complexity, introduce data augmentation, and use a learning rate scheduler to push for higher accuracy.
- **Specific Adjustments**:
    1.  **Data Augmentation**: Added `RandomCrop` and `RandomHorizontalFlip` to the training set.
    2.  **Increased Model Depth and Width**:
        - Increased the number and channel count of convolutional layers (`3 -> 32 -> 64 -> 128`).
        - Introduced `BatchNorm2d` to stabilize training.
        - Added `Dropout(0.5)` to prevent overfitting.
    3.  **Training Strategy**: Increased epochs to `30` and used a `StepLR` learning rate scheduler.
- **Results**:
    - **Accuracy**: 79%
    - **Parameters**: 357,258
    - **Score**: 2.2491
- **Analysis**: Accuracy improved significantly, but the excessive increase in parameters led to a lower final score. This confirmed that model complexity is a critical negative factor for the score.

#### Tuning Round 2: Pursuing Higher Efficiency (Optimal Solution)

- **Strategy**: Based on Round 1, drastically reduce the size of the fully connected layers, which contribute most to the parameter count, sacrificing a small amount of accuracy for a higher score.
- **Specific Adjustments**:
    - Kept the convolutional layers and data augmentation strategy.
    - Reduced the output dimension of the first fully connected layer from `128` to `64`.
- **Results**:
    - **Accuracy**: 73%
    - **Parameters**: 225,482
    - **Score**: **3.3050**
- **Analysis**: Although accuracy was lower than in Round 1, the substantial reduction in parameters resulted in a significant score improvement. This was the best result of the tuning process.

#### Tuning Round 3: Finding a Balance

- **Strategy**: Moderately increase the complexity of the fully connected layers from Round 2, attempting to boost accuracy without a major parameter increase.
- **Specific Adjustments**:
    - Changed the fully connected architecture to `Linear(..., 128)` -> `Linear(128, 64)` -> `Dropout` -> `Linear(64, 10)`.
- **Results**:
    - **Accuracy**: 81%
    - **Parameters**: 364,874
    - **Score**: 2.2497
- **Analysis**: Achieved the highest accuracy, but also the highest parameter count. The final score was similar to Round 1 and much lower than Round 2.

### Final Conclusion

The model from the second tuning round achieved the best performance under the competition's scoring criteria. Therefore, the final submitted code and model (`cifarNet.pth`) are the results of the second tuning round.

### How to Verify and Reproduce

1.  **Train the Model**:
    Ensure `cifar10Classification.py` is the version from **Tuning Round 2**. Run the following command to train and save the CIFAR-10 model.
    ```bash
    python3 "/Users/calvinlaam/Documents/DDMcourse/Spring/5055/Homework Part I/cifar10Classification.py"
    ```

2.  **Verify Score**:
    Run the official verification script to calculate the final score.
    ```bash
    python3 "/Users/calvinlaam/Documents/DDMcourse/Spring/5055/Homework Part I/accurateChecker3.py"
    ```
    The expected output should show an accuracy of approximately 73% and a final score of **3.3050**.

### Key Hyperparameters of the Final Model

- **Learning Rate (`lr`)**: `1e-3` (with a `StepLR` scheduler)
- **Number of Epochs (`epochs`)**: `30`
- **Optimizer**: `Adam`
- **Data Augmentation**: `RandomCrop`, `RandomHorizontalFlip`
- **Network Architecture**:
    - `Conv2d(3, 32)` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d`
    - `Conv2d(32, 64)` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d`
    - `Conv2d(64, 128)` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d`
    - `Linear(128 * 4 * 4, 64)` -> `ReLU` -> `Dropout(0.5)`
    - `Linear(64, 10)`
