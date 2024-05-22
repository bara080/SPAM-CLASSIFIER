############################################
#
#  Data Science:  Email Classifier
#
#  Written By : BARA AHMAD MOHAMMED
#
#############################################

# TODO : IMPORT ALL NEEDED CLASSIFIER
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Ensure the 'images' directory exists
os.makedirs('images', exist_ok=True)

# Fetch dataset
spambase = fetch_ucirepo(id=94)

# Data (as pandas dataframe)
X = spambase.data.features
y = np.ravel(spambase.data.targets)  # Reshape y into a 1d array


print(X.columns)
# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

# Multinomial Naive Bayes classifier
haram_bayes = MultinomialNB()
haram_bayes.fit(X_train, y_train)

# Evaluating the model
train_accuracy = haram_bayes.score(X_train, y_train)
test_accuracy = haram_bayes.score(X_test, y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Confusion Matrix
y_pred = haram_bayes.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Spam Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ['Non-Spam', 'Spam'])
plt.yticks([0, 1], ['Non-Spam', 'Spam'])
plt.savefig('images/confusion_matrix.png')
plt.show()

# ROC Curve
y_pred_prob = haram_bayes.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('images/ROC.png')
plt.show()
