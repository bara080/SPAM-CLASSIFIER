# SPAM-CLASSIFIER

# Data Science: Email Classifier

This project involves building an email classifier using the Multinomial Naive Bayes algorithm to classify emails as spam or non-spam (ham). It fetches the Spambase dataset from the UCI Machine Learning Repository, splits it into training and testing sets, trains a Multinomial Naive Bayes classifier, and evaluates its performance using various metrics.

## Project Overview

The email classifier project includes the following steps:

1. **Data Fetching**: Fetching the Spambase dataset from the UCI Machine Learning Repository using the `fetch_ucirepo` function.

2. **Data Preprocessing**: Splitting the dataset into features (X) and target labels (y), and further dividing them into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.

3. **Model Training**: Training a Multinomial Naive Bayes classifier using the `MultinomialNB` class from `sklearn.naive_bayes`.

4. **Model Evaluation**: Evaluating the performance of the classifier using accuracy scores, confusion matrix, and ROC curve.

## Setup

To set up the project environment, ensure you have the necessary dependencies installed. You can install them using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib ucimlrepo



## Setup

To execute the email classifier code, run the provided Python script (email.py) in your preferred Python environment. Ensure that you have the required datasets accessible.


## Output

Upon execution, the script will output the following:

Train accuracy: The accuracy of the classifier on the training set.
Test accuracy: The accuracy of the classifier on the testing set.
Confusion matrix: A matrix showing the true positives, true negatives, false positives, and false negatives.
ROC curve: A graphical representation of the Receiver Operating Characteristic (ROC) curve.
## License

This project is licensed under the MIT License. See the LICENSE file for details.
