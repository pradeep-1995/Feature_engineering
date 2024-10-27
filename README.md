---
# Diabetes Prediction Model

## Overview

This repository contains a machine learning model for predicting diabetes outcomes using the Pima Indians Diabetes Database. The model is built using TensorFlow and Keras, and it employs hyperparameter tuning through Keras Tuner to optimize performance.

## Table of Contents

- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [License](#license)

## Dataset

The dataset used in this project is the **Pima Indians Diabetes Database**, which can be found [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database). It consists of various medical predictor variables and one target variable, `Outcome`, which indicates whether the patient has diabetes (1) or not (0).

The dataset includes the following features:
- **Pregnancies:** Number of times pregnant
- **Glucose:** Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressure:** Diastolic blood pressure (mm Hg)
- **SkinThickness:** Triceps skin fold thickness (mm)
- **Insulin:** 2-Hour serum insulin (mu U/ml)
- **BMI:** Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction:** Diabetes pedigree function
- **Age:** Age (years)
- **Outcome:** Class variable (0 or 1)

## Technologies Used

- Python 3
- Pandas
- NumPy
- TensorFlow
- Keras
- Keras Tuner
- Scikit-learn

## Installation

To set up this project locally, clone the repository and install the required packages. You can create a virtual environment to manage dependencies.

```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt
```

You can create a `requirements.txt` file with the following content:

```
numpy
pandas
tensorflow
keras
keras-tuner
scikit-learn
```

## Usage

You can run the Jupyter notebook directly in your local environment or on platforms like Google Colab. The notebook contains all the steps required to preprocess the data, train the model, and perform hyperparameter tuning.

## Model Training

The model is a simple feedforward neural network with the following architecture:

- Input Layer: 8 features
- Hidden Layer(s): Variable number of layers and nodes based on hyperparameter tuning
- Output Layer: 1 node with sigmoid activation for binary classification

The model is compiled with:
- **Optimizer:** Adam or selected optimizer
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy

## Hyperparameter Tuning

Hyperparameter tuning is performed using Keras Tuner to optimize:
- Optimizer selection
- Number of nodes in hidden layers
- Number of hidden layers
- Dropout rates
- Activation functions

The best hyperparameters are determined based on validation accuracy.

## Results

After training the model and performing hyperparameter tuning, the following results were achieved (sample output):

- **Final Validation Accuracy:** Approximately 79.22%
- **Best Optimizer:** RMSprop
- **Best Number of Hidden Layers:** 4
- **Best Number of Units in Hidden Layer:** 85

These results may vary depending on the random state and the specific configuration used during tuning.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---
