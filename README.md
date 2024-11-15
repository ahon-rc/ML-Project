# CNN Model for Symbol Recognition

This repository contains an implementation of a basic Convolutional Neural Network (CNN) model designed to classify grayscale symbols. The model is trained and evaluated using PyTorch.

## Approach

### Data Preprocessing

1. **Normalization**: The training data is normalized to the range \([-1, 1]\) using PyTorch's `transforms.Normalize`.
2. **Tensor Conversion**: The dataset, consisting of grayscale images, is loaded and converted into tensors. Its unsqueezed to add an extra dimension as the channel.
3. **Custom Dataset Class**: A custom `SymbolDataset` class was created to handle data loading and indexing as I had trouble with the `TensorDataset` Class.

### Model Architecture

The CNN architecture is composed of:

-   `Conv2d(1, 32, 3, padding=1) -> MaxPool2d(2, 2)`
-   `Conv2d(32, 64, 3, padding=1) -> MaxPool2d(2, 2)`
-   Fully connected layers: `64 * 7 * 7 -> 120 -> 15`

### Training

-   **Optimizer**: Adam optimizer was used with a learning rate of 0.001.
-   **Loss Function**: Cross-entropy loss (`nn.CrossEntropyLoss`) was employed.
-   **Hyperparameters**:
    -   Epochs: 5
    -   Batch size: 10

### Validation

1. **Train-Validation Split**: The dataset was split into 80% training and 20% validation.
2. **Cross-Validation**: A 5-fold cross-validation approach was used to evaluate the model's performance on unseen data.

### Metrics

-   **Accuracy**: Calculated as the percentage of correctly predicted labels.
-   **Validation Loss**: Monitored during training to track performance.
-   **RMSE (Root Mean Square Error)**: Computed for additional evaluation of the final predictions.

### Final Accuracy

-   After training and validation, the final accuracy of the network was reported as **99.625%**.
-   The RMSE score across folds was **0.18264590203762054**.

---
