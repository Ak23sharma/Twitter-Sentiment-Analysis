# Twitter Sentiment Analysis Using Machine Learning

## 📌 Overview

This project uses Machine Learning and Natural Language Processing (NLP) to classify tweets as **Positive or Negative** based on their textual content.

The model is trained on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140), which contains 1.6 million tweets labeled according to their sentiment. It uses text preprocessing, stemming, TF-IDF vectorization, and Logistic Regression to perform sentiment classification.

## 🎯 Features

- Analyzes tweet text to identify sentiment.
- Preprocesses text using lowercasing, special character removal, stopword removal, and stemming.
- Converts text into numerical features using TF-IDF.
- Trains a Logistic Regression classification model.
- Evaluates performance using accuracy scores on training and testing data.
- Saves the trained model using Pickle for future predictions.
- Supports predictions for individual tweets and batches of tweets.

## 🛠️ Technologies Used

- **Python** — Core programming language
- **Pandas** — Dataset loading and manipulation
- **NumPy** — Numerical operations
- **NLTK** — Stopword removal and stemming
- **Regular Expressions (re)** — Text cleaning
- **Scikit-learn** — TF-IDF vectorization, dataset splitting, Logistic Regression, and model evaluation
- **Pickle** — Saving and loading the trained model
- **Google Colab** — Development environment
- **Kaggle API** — Dataset download

## 📂 Dataset

**Dataset:** [Sentiment140 – Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

The dataset contains approximately 1.6 million tweets with six columns:

| Column | Description |
|---|---|
| `target` | Sentiment label: `0` for negative and `4` for positive |
| `id` | Tweet identifier |
| `date` | Date and time of the tweet |
| `flag` | Query or search information |
| `user` | Twitter username |
| `text` | Tweet content |

For binary classification, the original positive label `4` is converted to `1`.

- `0` → Negative
- `1` → Positive

## ⚙️ Project Workflow

1. **Dataset Collection:** Download the Sentiment140 dataset from Kaggle.
2. **Data Preprocessing:** Load the dataset, assign column names, and inspect missing values and sentiment distribution.
3. **Text Cleaning:** Remove non-alphabetic characters and convert text to lowercase.
4. **Stopword Removal:** Remove common English words using NLTK.
5. **Stemming:** Reduce words to their stem forms using the Porter Stemmer.
6. **Feature Extraction:** Convert preprocessed tweets into numerical representations using TF-IDF.
7. **Data Splitting:** Split the dataset into 80% training data and 20% testing data, maintaining the class distribution.
8. **Model Training:** Train a Logistic Regression classifier on the TF-IDF features.
9. **Model Evaluation:** Calculate training and testing accuracy.
10. **Model Persistence:** Save the trained model using Pickle and load it for future predictions.

## 🧠 Machine Learning Model

**Algorithm:** Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for classification. In this project, it learns patterns from labeled tweets and predicts whether a tweet expresses positive or negative sentiment.

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts text into numerical features based on the importance of words within tweets and across the dataset.

## 📊 Results

- **Dataset size:** Approximately 1.6 million tweets
- **Training/testing split:** 80% / 20%
- **Reported model accuracy:** Approximately 77.6%
- **Evaluation metric:** Accuracy score

*Note: The reported accuracy is based on the project's recorded results. Run the notebook to reproduce and verify the exact training and testing scores.*

## 🚀 Getting Started

### Prerequisites

Install Python and the required libraries:

```bash
pip install numpy pandas nltk scikit-learn
```

### 1. Download the Dataset

Download the Sentiment140 dataset from Kaggle and extract the CSV file.

If you use the Kaggle API, configure your Kaggle API credentials according to the [Kaggle API documentation](https://github.com/Kaggle/kaggle-api).

### 2. Load the Dataset

```python
import pandas as pd

columns_names = [
    "target", "id", "date", "flag", "user", "text"
]

twitter_dataset = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    names=columns_names,
    encoding="ISO-8859-1"
)

twitter_dataset["target"] = twitter_dataset["target"].replace(4, 1)
```

### 3. Run the Notebook

Open `TwitterSentimnetAnalysis.ipynb` in Google Colab or Jupyter Notebook and execute the cells in order to preprocess the dataset, train the model, and evaluate its performance.

## 💾 Model Saving and Loading

The trained Logistic Regression model is saved as `training_model.sav`.

```python
import pickle

with open("training_model.sav", "wb") as file:
    pickle.dump(model, file)

with open("training_model.sav", "rb") as file:
    loaded_model = pickle.load(file)
```

**Important:** To predict sentiment for new, raw tweets, save and reuse the fitted TF-IDF vectorizer alongside the model. The model expects numerical TF-IDF features, not raw text.

## 🔮 Example Prediction

After training the model and fitting the TF-IDF vectorizer:

```python
tweet = ["I really love this product"]

processed_tweet = [stemming(tweet[0])]
tweet_vector = vectorizer.transform(processed_tweet)

prediction = loaded_model.predict(tweet_vector)

if prediction[0] == 0:
    print("Negative Tweet")
else:
    print("Positive Tweet")
```

## 🔮 Future Improvements

- Build a web interface using Flask, Django, or Streamlit.
- Save the TF-IDF vectorizer alongside the trained model.
- Evaluate the model using precision, recall, F1-score, and a confusion matrix.
- Improve text preprocessing to preserve useful sentiment signals, such as negations and emoticons.
- Compare Logistic Regression with Naive Bayes, Support Vector Machines, and other classification algorithms.
- Deploy the model as an API for real-time sentiment predictions.

## 👨‍💻 Author

Developed as a Machine Learning and Natural Language Processing project to explore sentiment classification using Python.

## 📄 License

This project is intended for educational and learning purposes. Refer to the original dataset's terms for usage and redistribution requirements.
