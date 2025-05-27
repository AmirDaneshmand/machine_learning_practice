# 🧠 Fraud Detection using Self-Organizing Maps (SOM)

This Jupyter Notebook demonstrates how to use a **Self-Organizing Map (SOM)** for **unsupervised anomaly detection** on credit card applications.

## 📌 Project Overview

- **Goal**: Detect potentially fraudulent applications using SOM
- **Technique**: Unsupervised learning with dimensionality reduction and visualization
- **Tools**: Jupyter Notebook, Python, MiniSom

---

## 📂 Files Included

| File                            | Description                                      |
|---------------------------------|--------------------------------------------------|
| `SOM_Fraud_Detection.ipynb`     | Main notebook with full implementation and results |
| `Credit_Card_Applications.csv`  | Dataset containing customer application details |

---

## 🚀 How to Run

### 1. Clone the repository and install dependencies:

```bash
pip install numpy pandas matplotlib minisom
```

2. Open the notebook:
jupyter notebook SOM_Fraud_Detection.ipynb

4. Follow the cells:
Data preprocessing

SOM training and visualization

Extracting anomalous (fraudulent) records

📈 SOM Visualization
The U-Matrix (distance map) helps detect anomalies:

Bright cells: high mean inter-neuron distance (potential outliers)

Markers: show approved (green) and rejected (red) applications

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()




📊 Dataset Information
The dataset contains various numerical and categorical features related to credit card applications.
The last column (Class) indicates approval status:

1 → Approved

0 → Rejected

We apply Min-Max scaling to normalize the data before feeding it to the SOM.

![image](https://github.com/user-attachments/assets/51d4d955-0c8e-4604-b5a5-a95436a7c372)

📌 Key Concepts
Self-Organizing Maps: A type of unsupervised neural network for dimensionality reduction and pattern recognition.

Unsupervised Learning: No labeled fraud data needed.

Anomaly Detection: SOM clusters are used to find patterns that differ from the majority.

🔍 Fraud Detection Strategy
Identify SOM nodes with high inter-neuron distances

Extract input vectors (customers) mapped to these nodes

Inverse-transform them to original scale for review

📚 Further Reading
Kohonen Self-Organizing Maps (Wikipedia)

MiniSom GitHub

Elbow Method for K-Means Clustering

👨‍💻 Author
This notebook was created as part of my Deep Learning A-Z 2025 course journey.
Feel free to fork, use, and build upon this project.

⭐ If you find this useful, please give the repo a star on GitHub!
