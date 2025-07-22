# Airline-Sentiment-Analysis-Using-RNN
# ✈️ Airline Tweets Sentiment Analysis using RNN

This project analyzes airline-related tweets to classify public sentiment as **positive**, **neutral**, or **negative** using a Recurrent Neural Network (RNN).

---

## 📊 Dataset

The dataset used is the publicly available **[Airline Tweets Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)**.

- **Size:** 14,640 tweets
- **Columns:** `text`, `airline`, `sentiment`, and more
- **Preprocessing Steps:**
  - Lowercasing
  - Removal of punctuation, links, usernames
  - Tokenization and padding
  - Label encoding for sentiment

---

## 🧠 Model Architecture

A deep learning model built using **TensorFlow/Keras** with the following structure:

- `Embedding` layer
- Two stacked `LSTM` layers with `Dropout`
- `Dense` output layer with `softmax` activation

**Training Setup:**
- Optimizer: `Adam`
- Loss Function: `Sparse Categorical Crossentropy`
- Epochs: 10

---

## 📈 Results & Evaluation

The model was evaluated using test data and key classification metrics:

| Metric        | Score |
|---------------|-------|
| Accuracy       | 66%   |
| Precision      | 66%   |
| Recall         | 99%   |
| F1-Score       | 79%   |

**Visualizations Included:**
- Sentiment label distribution
- Training & validation accuracy/loss
- Confusion matrix
- Classification report heatmap

---

## 📁 Repository Contents

- `airline_sentiment_rnn.ipynb` – Main notebook with all code and visualizations
- `Tweets.csv` – Input dataset (from Kaggle)
- `requirements.txt` – List of required Python packages
- `README.md` – This file

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook airline_sentiment_rnn.ipynb
