
# Email Spam Detection

This repository contains code for detecting spam emails using machine learning techniques. The dataset used is "spam.csv", and the code is written in Python using libraries like pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, wordcloud, and Streamlit.

## Getting Started

### Prerequisites

Make sure you have the following installed on your local machine:

- Python 
- VS Code
- Required Python libraries: numpy, pandas, scikit-learn, nltk, matplotlib, seaborn, wordcloud, Streamlit

### Installation

1. Clone the repository:

    ```
    git clone https://github.com/your_username/      email-spam-detection.git
    ```

2. Navigate to the project directory:

    ```
    cd email-spam-detection
    ```

3. finally run:

    ```
    code ./
    ```
    


## Documentation

### Data

The **data** section describes the dataset used in the project, its structure, and any preprocessing steps applied.

The dataset used for this project is **"spam.csv"**, which contains email messages labeled as spam or non-spam (ham). The dataset consists of two columns: **"target"** (label) and **"text"** (email content). Before training the model, the dataset undergoes preprocessing steps such as 
- cleaning
- encoding labels
- feature extraction

## Data Preprocessing

Preprocessing steps include:

- Lowercasing
- Tokenization
- Removing special characters
- Removing stop words and punctuation
- Stemming

### Exploratory Data Analysis (EDA)

The EDA section explains the exploratory data analysis performed on the dataset, including visualizations and insights gained from the data.

**Exploratory Data Analysis (EDA)** includes analyzing the distribution of characters, words, and sentences in spam and ham emails, visualizing word clouds, and exploring correlations between features.

### Model Building

The model building section discusses the various **machine learning algorithms** used for building the spam detection model, as well as their performance evaluation.

Various **machine learning** algorithms are used for model building, including 
-  Gaussian Naive Bayes
-  Multinomial Naive Bayes
-  Bernoulli Naive Bayes
-  Support Vector Classifier (SVC)
-  K-Nearest Neighbors (KNN)
-  Decision Tree Classifier (DTC)
-  Logistic Regression (LR)
-  Random Forest Classifier (RFC)
-  AdaBoost Classifier, Bagging Classifier
- Extra Trees Classifier (ETC)
-  Gradient Boosting Classifier (GBDT)
-  XGBoost Classifier (XGB).

**Model performances are evaluated based on accuracy and precision scores, and the best-performing algorithms are identified.**

## Project Files

### Notebooks and Scripts

- [email_spam_detection.ipynb](email_spam_detection.ipynb): Jupyter Notebook containing the code for email spam detection.
- [index.py](index.py): Python script for the Streamlit application.
- [model.pkl](model.pkl): Pickled machine learning model for spam detection.
- [vectorizer.pkl](vectorizer.pkl): Pickled vectorizer used for text vectorization.

### Data and Dependencies

- [spam.csv](spam.csv): Dataset containing email messages labeled as spam or ham.
### `nltk.txt`

- [nltk.txt](nltk.txt) contains the NLTK (Natural Language Toolkit) dependencies required for text processing:
  - `stopwords`: NLTK module for handling common stopwords.
  - `punkt`: NLTK module for tokenization.

### `requirements.txt`

- [requirements.txt](requirements.txt) specifies the main project dependencies required for running the code and the Streamlit application:
  - `streamlit`: Streamlit library for building interactive web applications.
  - `nltk`: Natural Language Toolkit for text processing tasks.
  - `sklearn`: Scikit-learn library for machine learning algorithms and utilities.




