# 🐶🐱 Cat vs Dog Classifier - CNN with Keras (.ipynb)

This Jupyter Notebook project demonstrates a convolutional neural network (CNN) for classifying images of cats and dogs using TensorFlow and Keras.

## 📌 Overview

- **Type:** Binary Image Classification (Dog vs. Cat)
- **Model:** Sequential CNN (2 conv layers + dense)
- **Data Augmentation:** Yes
- **Regularization:** Dropout
- **Callbacks:** EarlyStopping
- **Accuracy:** ~91% (train), ~85% (validation)

## 📁 Dataset (Not Included)

The dataset used is the [Dogs vs. Cats dataset from Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). Due to size limitations, the dataset is **not uploaded** to this repository.

**Expected folder structure:**

dataset/
├── training_set/
│ ├── cats/
│ └── dogs/
└── test_set/
├── cats/
└── dogs/


🧪 Model Evaluation
✅ Uses training/validation accuracy and loss plots

✅ Displays predictions on custom input images

✅ Shows prediction confidence

Prediction: dog (99.89% confidence)
Prediction: cat (59.41% confidence)


💾 Saving & Loading Model
The trained model is saved as:

Copy
Edit
cat_dog_classifier_model.h5
You can later load it for predictions without retraining.

🔮 Future Enhancements
Use transfer learning with VGG16 or MobileNet

Add GUI using Streamlit or Gradio

Run on Google Colab for GPU acceleration

📸 Screenshots



![image](https://github.com/user-attachments/assets/eaebb8c2-2b9d-4c9f-b419-f83c6962c5ca)




