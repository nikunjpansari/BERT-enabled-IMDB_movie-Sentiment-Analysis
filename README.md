# IMDB Movie Review Sentiment Analysis

A Streamlit application that leverages a fine-tuned BERT model to analyze the sentiment of IMDB movie reviews. The app not only classifies reviews as **Positive** or **Negative**, but also provides detailed metrics such as class probabilities and token count.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

IMDB Movie Review Sentiment Analysis is an intuitive web app built using [Streamlit](https://streamlit.io/) and [TensorFlow](https://www.tensorflow.org/) with a fine-tuned BERT model. The application allows users to input a movie review and instantly obtain a sentiment prediction along with detailed performance metrics.

## Features

- **Real-Time Sentiment Prediction:** Classify reviews into positive or negative sentiments using a fine-tuned BERT model.
- **Detailed Metrics:** In addition to the overall prediction, the app displays:
  - **Overall Confidence:** Confidence level of the predicted sentiment.
  - **Class Probabilities:** Individual probabilities for the positive and negative classes.
  - **Token Count:** The number of tokens present in the review.
- **Modern UI:** Attractive and responsive interface with an external CSS file for unique styling.
- **Multi-Page Navigation:** Easy navigation between the sentiment analysis page and an informative "About" page.
- **Sample Reviews:** Preloaded sample reviews to quickly test and demonstrate the app's functionality.

## Installation

### Prerequisites

- Python 3.7+
- [pip](https://pip.pypa.io/en/stable/)

## Clone the Repository

```shell
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

## Install Required Packages

```shell
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


### Install the necessary dependencies:

```shell
pip install -r requirements.txt

### Usage
To run the application, execute the following command:

``shell
streamlit run app.py
