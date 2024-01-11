# Twitter Sentiment Analysis

## Overview

This project focuses on sentiment analysis of Twitter data using Natural Language Processing (NLP) techniques and machine learning. The objective is to classify tweets into sentiment categories, including positive, negative, or neutral. The project leverages popular Python libraries such as pandas, scikit-learn, nltk, and regex for data manipulation, preprocessing, and model building.

## Key Features

- **Data Exploration**: Utilizes pandas for loading and exploring the Twitter sentiment dataset, providing insights into the data's structure and content.

- **Text Preprocessing**: Implements a comprehensive text preprocessing pipeline, including lowercase conversion, text cleaning, tokenization, stop-word removal, and lemmatization to enhance the quality of textual data.

- **TF-IDF Vectorization**: Applies the Term Frequency-Inverse Document Frequency (TF-IDF) technique to convert raw text into numerical feature vectors, capturing the significance of words in the dataset.

- **k-Nearest Neighbors (k-NN) Classifier**: Utilizes the k-NN algorithm from scikit-learn for sentiment classification, leveraging the similarity of neighboring data points to make predictions.

- **Train-Test Split**: Employs scikit-learn's `train_test_split` function to divide the dataset into training and testing sets, ensuring a robust evaluation of the sentiment analysis model.

- **Model Evaluation**: Utilizes accuracy as the primary evaluation metric, providing an indication of the model's overall performance on the test data.

## Usage

1. **Install Dependencies**: Ensure the required Python libraries (`pandas`, `regex`, `numpy`, `scikit-learn`, `nltk`) are installed.

    ```bash
    pip install pandas regex numpy scikit-learn nltk
    ```

2. **Download Dataset**: https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset

3. **Run the Script**: Execute the provided Python script, adjusting file paths if necessary.

    ```bash
    python sentiment_analysis.py
    ```

## Contributing

Contributions are welcomed! If you encounter issues or have suggestions for enhancements, feel free to open an issue or submit a pull request.



