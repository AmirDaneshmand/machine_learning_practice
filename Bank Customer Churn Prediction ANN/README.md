# 🧠 Bank Customer Churn Prediction using Neural Networks

This project predicts customer churn (i.e., whether a bank customer will leave or not) using an artificial neural network (ANN) built with TensorFlow/Keras. The dataset includes demographic and financial details of 10,000 bank customers.

---

## 📊 Dataset

The dataset used is `Churn_Modelling.csv` with 14 columns:
- Demographic info: `Geography`, `Gender`, `Age`
- Financial info: `CreditScore`, `Balance`, `EstimatedSalary`, etc.
- Target variable: `Exited` (1 if customer left, 0 otherwise)

---

## 🔧 Preprocessing Steps

- Dropped unnecessary columns: `RowNumber`, `CustomerId`, `Surname`
- Label encoded `Gender`
- One-hot encoded `Geography`
- Feature scaling using `StandardScaler`
- Train-test split (80/20)

---

## 🧠 Model Architecture

A Sequential ANN with the following structure:
- 4 hidden layers with 6 neurons each, `ReLU` activation
- Output layer with 1 neuron, `Sigmoid` activation
- Optimizer: `Adam`
- Loss function: `Binary Crossentropy`
- Trained for 100 epochs with batch size 32

---

## 📈 Evaluation Metrics

Used:
- Accuracy Score
- Classification Report
- Confusion Matrix

### 🧪 Results:
Accuracy: 86.25%

Precision / Recall / F1-score for class 1 (churn):

Precision: 72%

Recall: 53%

F1-score: 61%



---

## 📦 Libraries Used

- pandas, numpy, seaborn, matplotlib
- sklearn
- tensorflow / keras

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python script

---

## 💡 Future Improvements

- Use GridSearchCV or KerasTuner for hyperparameter tuning
- Handle class imbalance using SMOTE or class weights
- Experiment with deeper or regularized architectures

---

## 📁 License

This project is open-source and available under the MIT License.

---

## ✨ Author

Created by [AmirReza Daneshmand](https://github.com/AmirDaneshmand)
