# 📧 Spam Classification Model Evaluation

This project evaluates multiple machine learning and deep learning models to classify SMS messages as spam or not spam. The dataset is preprocessed into a bag-of-words representation with over **1,300 word-frequency features** and a binary `label` column (0 = ham, 1 = spam).

## 📊 Project Highlights

🧠 Models Used
- **Multi-Layer Perceptron (MLP)** (using TensorFlow/Keras)
- **Decision Tree Classifier (DT)**
- **Random Forest Classifier (RF)**
- **Support Vector Classifier (SVM)**

📊 Metrics Evaluated
Each model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Log Loss**

## 🧠 Key Findings

Based on the evaluation results:
- MLP and SVM were the top performers, each achieving 98.7% accuracy, 0.993 precision, 0.913 recall, and an F1 score of 0.951.
  - MLP slightly outperformed SVM in log loss: 0.0532 vs. 0.0599.
- Random Forest also performed well:
  - 97.1% accuracy, 0.964 precision, 0.832 recall, 0.893 F1 score, but with a higher log loss of 0.4045.
- Decision Tree had the weakest results:
  - 90.9% accuracy, 0.698 precision, 0.646 recall, 0.671 F1 score, and 0.2347 log loss.
➡️ Conclusion: The MLP model offers the best balance of predictive accuracy and probabilistic confidence, making it the most suitable for this classification task.


## 📂 Dataset

- Dataset files: `train.csv` and `test.csv` (already included in this repository).
- Each row represents frequency counts for 1364 unique words, plus a binary `label` column indicating spam (1) or ham (0).

## 📁 Files Included

- `Potri_Abhisri_Barama_Spam_Classification_Model_Evaluation.ipynb`: Main analysis and evaluation notebook  
- `train.csv`, `test.csv`: Labeled dataset split for model training and evaluation

## 🛠 Tools & Libraries
This project uses the following Python libraries:
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib
- **Machine Learning Models:** scikit-learn (DecisionTreeClassifier, RandomForestClassifier, SVC)
- **Deep Learning:** TensorFlow, Keras (Sequential, Dense, Dropout)
- **Model Evaluation:** accuracy_score, precision_score, recall_score, f1_score, log_loss
- **Optimization:** GridSearchCV, KFold

> 📌 All dependencies are already imported in the notebook. No additional setup required.

## 🚀 How to Run

1. Clone Repository
2. Launch Jupyter Notebook
3. Run All Cells
