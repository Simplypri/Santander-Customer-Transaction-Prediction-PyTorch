Santander Customer Transaction Prediction

This project implements a deep learning solution for predicting customer transactions using the Santander Customer Transaction Dataset. The task is a binary classification problem, where the goal is to predict whether a customer will make a specific transaction based on anonymized features.

**Dataset Overview**

Training data: ~200,000 samples, 200 numerical features (var_0 to var_199), ID_code, and target column. Test data: Similar structure without the target.

Target distribution: Highly imbalanced (≈90% negative class, ≈10% positive class). The dataset is commonly used in Kaggle competitions and reflects real-world imbalanced binary classification scenarios.

**Project Workflow**

1. Data Exploration & Preprocessing

2. Load CSVs into pandas DataFrames.

3. Inspect data types, distributions, and feature ranges.

4. Separate features (X) and target (y) for training.

5. Standardize features using StandardScaler.

6. PyTorch Dataset & DataLoader

7. Create custom SantanderDataset class for training and test data.

8. Define DataLoader for batching and shuffling.

**Neural Network Design**

SantanderNet: Fully connected network (200 → 256 → 128 → 1). Includes batch normalization, ReLU activations, and dropout (0.3). Uses BCEWithLogitsLoss for stable binary cross-entropy. Optimizer: Adam with learning rate 0.001.

**Training & Evaluation**

Track training loss and ROC–AUC score per epoch. Log metrics using TensorBoard (SummaryWriter). Apply early stopping based on convergence (ΔLoss < 0.001). 

**Hyperparameter Tuning**

Experiment with learning rate, optimizer, dropout, and epochs. Best configuration: Adam, lr=0.001, dropout=0.3 → AUC ≈ 0.9251.

**Handling Class Imbalance**

Dataset is highly imbalanced (179,902 class 0 vs 20,098 class 1). Techniques to improve minority-class learning:

Class weighting in loss function

Oversampling / undersampling

SMOTE (synthetic data generation)

Trade-offs include overfitting or information loss.

**Results**

Model achieved high ROC–AUC (≈0.980) on training data.

Confusion matrix indicates good performance for majority class; minority class predictions are fewer due to class imbalance.

Early stopping ensures efficient training without overfitting.

**Technologies Used**

PyTorch – deep learning framework

Pandas / NumPy – data manipulation

Scikit-learn – preprocessing, evaluation metrics

Matplotlib / Seaborn – visualization

TensorBoard – training metric logging

Jupyter Notebook – experimentation

**References**

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.

He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE TKDE.

Han, H., et al. (2005). Borderline-SMOTE.
