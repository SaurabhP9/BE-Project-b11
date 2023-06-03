# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pickle

# # Read CSV files and concatenate into one DataFrame
# df1 = pd.read_csv(
#     "D:/Final-Year-Project-master/comments_commedy.csv", encoding="iso-8859-1"
# )
# df1.drop(["Value.videoId", "Name", "Value.author"], axis=1, inplace=True)
# df1["Category"] = "Comedy"

# df2 = pd.read_csv(
#     "D:/Final-Year-Project-master/comments_science1.csv", encoding="iso-8859-1"
# )
# df2.drop(["Value.videoId", "Name", "Value.author"], axis=1, inplace=True)
# df2["Category"] = "Science"

# df3 = pd.read_csv("D:/Final-Year-Project-master/comments_tv.csv", encoding="iso-8859-1")
# df3.drop(["Value.videoId", "Name", "Value.author"], axis=1, inplace=True)
# df3["Category"] = "TV"

# df4 = pd.read_csv(
#     "D:/Final-Year-Project-master/comments_news.csv", encoding="iso-8859-1"
# )
# df4.drop(["Value.videoId", "Name", "Value.author"], axis=1, inplace=True)
# df4["Category"] = "News"

# df5 = pd.read_csv(
#     "D:/Final-Year-Project-master/comments_science2.csv", encoding="iso-8859-1"
# )
# df5.drop(["Value.videoId", "Name", "Value.author"], axis=1, inplace=True)
# df5["Category"] = "Science"

# df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
# print("Values Count: " + str(df["Category"].value_counts()))

# # Preprocess the text data
# df["text"] = df["text"].str.lower()
# df["text"] = df["text"].str.replace("[^\w\s]", "")
# df["text"] = df["text"].str.replace("\d+", "")
# df.dropna(subset=["text"], inplace=True)
# df = df[~df.isin([" ", "  "]).any(axis=1)]
# stop = set(stopwords.words("english"))
# ps = SnowballStemmer("english")
# df["text"] = df["text"].apply(
#     lambda x: " ".join([ps.stem(word) for word in x.split() if word not in (stop)])
# )

# # Convert text data to TF-IDF vectors
# vectorizer = CountVectorizer()
# tfidf_transformer = TfidfTransformer(use_idf=True)

# comment = df["text"].tolist()
# y_train = df["Category"].values

# X = vectorizer.fit_transform(comment)
# X_train = tfidf_transformer.fit_transform(X)

# # Save the vectorizer and trained model to disk
# with open("vectorizer_pickle", "wb") as file:
#     pickle.dump(vectorizer, file)
#     print("Saved vectorizer to disk")

# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# with open("model_pickle", "wb") as file:
#     pickle.dump(clf, file)
#     print("Saved model to disk")

# # Predict on the training set
# y_pred = clf.predict(X_train)

# # Calculate accuracy using confusion matrix
# cm = confusion_matrix(y_train, y_pred)
# print("Confusion Matrix:")
# print(cm)

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # Calculate accuracy
# accuracy = (cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3]) / cm.sum()
# print("Accuracy: {:.2f}%".format(accuracy * 100))


# ?????


# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pickle

# # Read CSV files and concatenate into one DataFrame
# df1 = pd.read_csv(
#     "D:\Final-Year-Project-master\Merged_commedy.csv", encoding="iso-8859-1"
# )
# df1.drop(["Value.videoId", "Value.author"], axis=1, inplace=True)
# df1["Category"] = "Comedy"

# df2 = pd.read_csv(
#     "D:\Final-Year-Project-master\Merged_science.csv", encoding="iso-8859-1"
# )
# df2.drop(["Value.videoId", "Value.author"], axis=1, inplace=True)
# df2["Category"] = "Science"

# df3 = pd.read_csv("D:\Final-Year-Project-master\Merged_tv.csv", encoding="iso-8859-1")
# df3.drop(["Value.videoId", "Value.author"], axis=1, inplace=True)
# df3["Category"] = "TV"

# df4 = pd.read_csv("D:\Final-Year-Project-master\Merged_news.csv", encoding="iso-8859-1")
# df4.drop(["Value.videoId", "Value.author"], axis=1, inplace=True)
# df4["Category"] = "News"

# df5 = pd.read_csv(
#     "D:/Final-Year-Project-master/comments_science2.csv", encoding="iso-8859-1"
# )
# df5.drop(["Value.videoId", "Value.author"], axis=1, inplace=True)
# df5["Category"] = "Science"

# df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
# print("Values Count: " + str(df["Category"].value_counts()))

# # Preprocess the text data
# df["text"] = df["text"].str.lower()
# df["text"] = df["text"].str.replace("[^\w\s]", "")
# df["text"] = df["text"].str.replace("\d+", "")
# df.dropna(subset=["text"], inplace=True)
# df = df[~df.isin([" ", "  "]).any(axis=1)]
# stop = set(stopwords.words("english"))
# ps = SnowballStemmer("english")
# df["text"] = df["text"].apply(
#     lambda x: " ".join([ps.stem(word) for word in x.split() if word not in (stop)])
# )

# # Convert text data to TF-IDF vectors
# vectorizer = CountVectorizer()
# tfidf_transformer = TfidfTransformer(use_idf=True)

# comment = df["text"].tolist()
# y_train = df["Category"].values

# X = vectorizer.fit_transform(comment)
# X_train = tfidf_transformer.fit_transform(X)

# # Save the vectorizer and trained model to disk
# with open("vectorizer_pickle", "wb") as file:
#     pickle.dump(vectorizer, file)
#     print("Saved vectorizer to disk")

# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# with open("model_pickle", "wb") as file:
#     pickle.dump(clf, file)
#     print("Saved model to disk")

# # Predict on the training set
# y_pred = clf.predict(X_train)

# # Calculate accuracy using confusion matrix
# cm = confusion_matrix(y_train, y_pred)
# print("Confusion Matrix:")
# print(cm)

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # Calculate accuracy
# accuracy = (cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3]) / cm.sum()
# print("Accuracy: {:.2f}%".format(accuracy * 100))

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Read CSV files and concatenate into one DataFrame
df1 = pd.read_csv(
    "D:\Final-Year-Project-master\Merged_commedy.csv", encoding="iso-8859-1"
)
df1.drop(["Value.videoId", "Value.author"], axis=1, inplace=True)
df1["Category"] = "Comedy"

df2 = pd.read_csv(
    "D:\Final-Year-Project-master\Merged_science.csv", encoding="iso-8859-1"
)
df2.drop(["Value.videoId", "Value.author"], axis=1, inplace=True)
df2["Category"] = "Science"

df3 = pd.read_csv("D:\Final-Year-Project-master\Merged_tv.csv", encoding="iso-8859-1")
df3.drop(["Value.videoId", "Value.author"], axis=1, inplace=True)
df3["Category"] = "TV"

df4 = pd.read_csv("D:\Final-Year-Project-master\Merged_news.csv", encoding="iso-8859-1")
df4.drop(["Value.videoId", "Value.author"], axis=1, inplace=True)
df4["Category"] = "News"

df5 = pd.read_csv(
    "D:/Final-Year-Project-master/comments_science2.csv", encoding="iso-8859-1"
)
df5.drop(["Value.videoId", "Value.author"], axis=1, inplace=True)
df5["Category"] = "Science"

df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
print("Values Count: " + str(df["Category"].value_counts()))

# Preprocess the text data
df["text"] = df["text"].str.lower()
df["text"] = df["text"].str.replace("[^\w\s]", "")
df["text"] = df["text"].str.replace("\d+", "")
df.dropna(subset=["text"], inplace=True)
df = df[~df.isin([" ", "  "]).any(axis=1)]
stop = set(stopwords.words("english"))
ps = SnowballStemmer("english")
df["text"] = df["text"].apply(
    lambda x: " ".join([ps.stem(word) for word in x.split() if word not in (stop)])
)

# Convert text data to TF-IDF vectors
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer(use_idf=True)

comment = df["text"].tolist()
y_train = df["Category"].values

X = vectorizer.fit_transform(comment)
X_train = tfidf_transformer.fit_transform(X)

# Save the vectorizer and trained model to disk
with open("vectorizer_pickle", "wb") as file:
    pickle.dump(vectorizer, file)
    print("Saved vectorizer to disk")

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

with open("model_pickle", "wb") as file:
    pickle.dump(clf, file)
    print("Saved model to disk")

# Predict on the training set
y_pred = clf.predict(X_train)

# Calculate accuracy using confusion matrix
cm = confusion_matrix(y_train, y_pred)
print("Confusion Matrix:")
print(cm)

# Print classification report
report = classification_report(y_train, y_pred)
print("Classification Report:")
print(report)

# Extract precision, recall, f1-score, and support from the classification report
target_names = df["Category"].unique()
classification_metrics = classification_report(
    y_train, y_pred, target_names=target_names, output_dict=True
)

# Print precision, recall, f1-score, and support for each category
for category in target_names:
    metrics = classification_metrics[category]
    print(f"\nMetrics for category: {category}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1-Score: {metrics['f1-score']}")
    print(f"Support: {metrics['support']}")

# Calculate overall precision, recall, f1-score, and support
overall_metrics = classification_metrics["weighted avg"]
print("\nOverall Metrics")
print(f"Overall Precision: {overall_metrics['precision']}")
print(f"Overall Recall: {overall_metrics['recall']}")
print(f"Overall F1-Score: {overall_metrics['f1-score']}")
print(f"Overall Support: {overall_metrics['support']}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Calculate accuracy
accuracy = (cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3]) / cm.sum()
print("\nAccuracy: {:.2f}%".format(accuracy * 100))
