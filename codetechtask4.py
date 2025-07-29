import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
print("Sample data:")
display(data.head())


data['label'].value_counts().plot(kind='bar', title='Class Distribution', xlabel='Label', ylabel='Count')
plt.show()


data['label'] = data['label'].map({'ham': 0, 'spam': 1})


X = data['message']
y = data['label']


vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f" Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print(" Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

def predict_message(msg):
    vect = vectorizer.transform([msg])
    result = model.predict(vect)[0]
    return "Spam" if result == 1 else "Not Spam"
test_message = "You have won $1000. Claim your reward now!"
print(f"\n🔮 Prediction for test message:\n'{test_message}' ➤ {predict_message(test_message)}")
