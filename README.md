# README

## Overview

This repository contains several machine learning scripts and configurations for various models, including Long Short-Term Memory (LSTM), Support Vector Machine (SVM), Artificial Neural Network (ANN), autoencoders, and transformers. These scripts are designed to preprocess data, train models, and evaluate their performance on different datasets.

## Files and Descriptions

### Python Scripts

- **autoencoder.py**: Contains the implementation of an autoencoder for unsupervised learning tasks.
- **cifar.py**: Script to train a model on the CIFAR-10 dataset using standard approaches.
- **modified_cifar.py**: Modified version of the CIFAR-10 training script with custom settings and enhancements.
- **transformer.py**: Implements a transformer model for various tasks such as natural language processing.

### Data Preprocessing and Model Training Logs

#### LSTM
- **lstm 20240518 -2.txt**:
    - Logs for training the LSTM model.
    - Key Metrics:
      - Test Accuracy: 99.89%
      - Training Accuracy: 92.97%
      - Validation Accuracy: 94.41%
- **lstm 20240518.txt**:
    - Detailed steps and logs for data loading, cleaning, encoding, and training the LSTM model.
    - Key Metrics:
      - Test Accuracy: 99.89%
      - Training Accuracy: 92.99%
      - Validation Accuracy: 95.85%

#### SVM
- **svm -2.txt**:
    - Logs for training the SVM model with pre-encoded datasets.
    - Key Metrics:
      - Test Accuracy: 2.45%
      - Training Accuracy: 2.45%
      - Validation Accuracy: 2.90%
- **svm.txt**:
    - Detailed steps and logs for data loading, cleaning, encoding, and training the SVM model.
    - Key Metrics:
      - Test Accuracy: 2.49%
      - Training Accuracy: 2.49%
      - Validation Accuracy: 2.90%

#### ANN
- **ann 20240518 -2.txt**:
    - Logs for training the ANN model with pre-encoded datasets.
    - Key Metrics:
      - Test Accuracy: 99.44%
      - Training Accuracy: 99.45%
      - Validation Accuracy: 97.56%
- **ann 20240518.txt**:
    - Detailed steps and logs for data loading, cleaning, encoding, and training the ANN model.
    - Key Metrics:
      - Test Accuracy: 99.40%
      - Training Accuracy: 99.41%
      - Validation Accuracy: 97.56%

## Usage

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Run the scripts:**
    - Ensure you have all necessary dependencies installed.
    - Execute the desired script using Python.
    ```bash
    python <script_name>.py
    ```

## Dependencies

- Python 3.8+
- TensorFlow
- PyTorch
- Scikit-learn
- Numpy
- Pandas
- Matplotlib

## Results

Each model script outputs various performance metrics including accuracy, precision, recall, F1 score, and AUC for training, testing, and validation datasets. These metrics are logged in the respective text files for detailed analysis.

## Contributions

Feel free to fork this repository, create feature branches, and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

For any issues or questions, please open an issue in the repository or contact the maintainer.
