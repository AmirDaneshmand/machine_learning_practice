# 📈 LSTM-Based Stock Price Prediction

This project uses a Long Short-Term Memory (LSTM) neural network to predict the opening price of Google stock based on historical data.

---

## 🔍 Problem Statement

Stock price prediction is a common time series forecasting problem. The aim of this project is to build a deep learning model that forecasts the **next day's opening price** of Google stock using historical **Open, High, Low, and Close** prices.

---

## 🛠️ Technologies Used

- Python 🐍
- TensorFlow / Keras for deep learning
- Pandas and NumPy for data processing
- Scikit-learn for scaling and evaluation
- Matplotlib (optional) for visualization

---

## 📂 Dataset

The dataset used is [`Google_Stock_Price_Train.csv`] which contains the following columns:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

**Note:** Commas in large numbers (like `1,008.64`) are handled using `thousands=','` in `pd.read_csv()`.

---

## 🧠 Model Architecture

```python
Sequential([
    LSTM(128, return_sequences=True, input_shape=(60, 4)),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(1)
])

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Batch Size: 32

Epochs: 150

EarlyStopping: with val_loss, patience of 10 epochs


📊 Model Evaluation
The model was evaluated using the following metrics:

Metric	Value
MSE	e.g., 86.2765
RMSE	e.g., 9.2885
MAE	e.g., 8.1043
R²	e.g., 0.6030

Note: Exact values may vary based on training-test split and preprocessing.


🔍 Future Improvements
Incorporating additional features like Volume or technical indicators

Trying GRU or Transformer-based architectures

Hyperparameter tuning using Keras Tuner or Optuna

Developed by [AmirReza Daneshmand]
