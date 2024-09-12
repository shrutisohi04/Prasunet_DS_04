# Project Title: Sentiment Analysis and Visualization of Social Media Data

## Overview
This project focuses on analyzing and visualizing sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands. The dataset contains social media posts, and the goal is to preprocess the text, predict sentiments, and visualize the results.

## Libraries and Modules Used

### Pandas
**Pandas** is a powerful library for data manipulation and analysis in Python. It provides data structures like DataFrame and Series, which are essential for handling tabular data.

### TextBlob
**TextBlob** is a simple library for processing textual data. It provides a consistent API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.

### Matplotlib
**Matplotlib** is a comprehensive library for creating static, animated, and interactive visualizations in Python. It offers a wide variety of plots, including line plots, bar plots, histograms, and scatter plots.

### NLTK
**NLTK (Natural Language Toolkit)** is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources.

### Scikit-learn
**Scikit-learn** is a machine learning library for Python that provides simple and efficient tools for data mining and data analysis. It includes various classification, regression, and clustering algorithms.

### Re
**Re (Regular Expressions)** is a module in Python that provides support for working with regular expressions, which can be used for searching, matching, and manipulating strings.

## Steps Involved

### 1. Data Loading
Load the social media dataset using Pandas.

### 2. Data Preprocessing
- **Text Cleaning**: Remove punctuation, numbers, and special characters.
- **Lowercasing**: Convert text to lowercase.
- **Tokenization**: Split the text into tokens (words).
- **Stopwords Removal**: Remove common words that do not contribute to sentiment (e.g., "and", "the").
- **Stemming**: Reduce words to their root form using the Porter Stemmer.

### 3. Feature Extraction
Use TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer from Scikit-learn to transform the preprocessed text into numerical vectors that can be used for machine learning.

### 4. Sentiment Prediction
- Use TextBlob to analyze the sentiment of the text. The sentiment polarity score is used to classify the sentiment as positive, negative, or neutral.
- Train a machine learning model (e.g., Random Forest Classifier) to predict the sentiment of the text.

### 5. Model Training
- Split the dataset into training and testing sets using Scikit-learn's `train_test_split`.
- Train the Random Forest Classifier on the training set and evaluate its performance on the testing set.

### 6. Sentiment Analysis
- Define a function to preprocess the text.
- Define a function to predict the sentiment of new text inputs.

### 7. Visualization
Use Matplotlib and Seaborn to create visualizations that show sentiment distribution and patterns in the dataset.

## Functions
- **preprocess_text(text)**: Cleans and preprocesses the text data.
- **get_sentiment(text)**: Uses TextBlob to determine the sentiment of the text.
- **predict_sentiment(text)**: Predicts the sentiment of the given text using the trained model and TF-IDF vectorizer.

## References
- [Pandas Documentation](https://pandas.pydata.org/)
- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Re Documentation](https://docs.python.org/3/library/re.html)
