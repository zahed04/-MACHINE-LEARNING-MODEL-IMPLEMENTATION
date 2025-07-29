# -MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY:CODETECH IT SOLUTIONS

NAME:ZAHED HUSSAIN

INTERN ID:CT04DZ1510

DOMAIN:PYTHON PROGRAMMING

DURATION:4 WEEKS

MENTOR:NEELA SANTOSH

DESCRIPTION:

This code implements a spam detection system using machine learning (ML) and natural language processing (NLP) techniques. It classifies SMS messages into "spam" or "not spam" (also known as "ham") using a Naive Bayes classifier, a popular algorithm for text classification.

📌 Description of the Code Workflow
Importing Libraries:

pandas and numpy are used for data manipulation.

matplotlib.pyplot and seaborn are used for data visualization.

sklearn provides tools for ML such as data splitting, vectorization, model training, and evaluation.

Loading the Dataset:

The SMS dataset is fetched directly from a GitHub URL using pandas.read_csv.

The file sms.tsv is a tab-separated file with two columns: label (ham or spam) and message (the SMS text).

display(data.head()) shows the first five rows to give an idea of the data structure.

Visualizing Class Distribution:

A bar plot shows the distribution of "ham" vs "spam" messages to understand class balance. This helps detect if the data is imbalanced.

Encoding Labels:

Text labels are converted to numeric form: 'ham' → 0, 'spam' → 1. This is necessary for model training since most ML models require numeric inputs.

Splitting Data:

X contains the message texts, while y holds the corresponding labels.

The dataset is split into training (80%) and testing (20%) sets using train_test_split.

Text Vectorization:

Text data must be converted to numeric format. CountVectorizer from sklearn transforms the text into a bag-of-words representation. Each unique word becomes a feature, and the values are word counts per message.

Model Training and Prediction:

A Multinomial Naive Bayes model (MultinomialNB) is used, ideal for count data like bag-of-words.

The model is trained on the training set and tested on the test set to make predictions.

Model Evaluation:

Metrics used to evaluate the model include:

Accuracy Score: Proportion of correct predictions.

Classification Report: Shows precision, recall, and F1-score.

Confusion Matrix: Visualized using a heatmap to show how many spam/ham messages were correctly or incorrectly classified.

Message Prediction Function:

A function predict_message() allows prediction on new messages. It transforms the input message using the same vectorizer and returns a spam/ham classification.

Example: "You have won $1000. Claim your reward now!" is tested and classified appropriately.

🛠️ Tools & Libraries Used
Tool/Library	Purpose
pandas	Reading and handling structured data (DataFrame operations).
numpy	Supporting numerical operations.
matplotlib	Creating visual plots.
seaborn	Enhanced data visualization (confusion matrix).
sklearn.model_selection	Splitting dataset into training and testing parts.
sklearn.feature_extraction.text.CountVectorizer	Converting text into numerical vectors.
sklearn.naive_bayes.MultinomialNB	Classification model suitable for text classification.
sklearn.metrics	Model evaluation using accuracy, confusion matrix, etc.

✅ Conclusion
This code provides a simple yet powerful demonstration of how spam messages can be detected using machine learning. By combining text processing with the Naive Bayes algorithm and performance evaluation, it builds a real-world application capable of automatically classifying messages.


OUTPUT:

<img width="638" height="90" alt="Image" src="https://github.com/user-attachments/assets/94567b9f-ffd8-4c20-9dc6-31605feca4b8" />
