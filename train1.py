import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import pickle
import chardet
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn import tree


# Read CSV files and concatenate into one DataFrame
df1 = pd.read_csv(
    "D:/Final-Year-Project-master/comments_commedy.csv",
    encoding="iso-8859-1",
    # errors="ignore",
)
df1.drop(["Value.videoId", "Name", "Value.author"], axis=1, inplace=True)
df1["Category"] = "Comedy"
# print(df1)

df2 = pd.read_csv(
    "D:/Final-Year-Project-master/comments_science1.csv",
    encoding="iso-8859-1",
    # errors="ignore",
)
df2.drop(["Value.videoId", "Name", "Value.author"], axis=1, inplace=True)
df2["Category"] = "Science"
# print(df2)

df3 = pd.read_csv(
    "D:/Final-Year-Project-master/comments_tv.csv",
    encoding="iso-8859-1",
    # errors="ignore",
)
df3.drop(["Value.videoId", "Name", "Value.author"], axis=1, inplace=True)
df3["Category"] = "TV"
# print(df3)

df4 = pd.read_csv(
    "D:/Final-Year-Project-master/comments_news.csv",
    encoding="iso-8859-1",
    # errors="ignore",
)
df4.drop(["Value.videoId", "Name", "Value.author"], axis=1, inplace=True)
df4["Category"] = "News"
# print(df4)

df5 = pd.read_csv(
    "D:/Final-Year-Project-master/comments_science2.csv",
    encoding="iso-8859-1",
    # errors="ignore",
)
df5.drop(["Value.videoId", "Name", "Value.author"], axis=1, inplace=True)
df5["Category"] = "Science"
# print(df5)

df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
print("Values Count: " + str(df["Category"].value_counts()))
# print(df)

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
vectoriser = CountVectorizer()
tf_idf_vectorizer = TfidfTransformer(use_idf=True)


comment = df["text"].tolist()
y_train = df.iloc[:, 1].values

X = vectoriser.fit_transform(comment)
print(X)
X_Train = tf_idf_vectorizer.fit_transform(X)
print(X_Train)

# Save the vectorizer and trained model to disk
with open("vectorizer_pickle", "wb") as file:
    pickle.dump(vectoriser, file)
    print("done vectoriser")

clf1 = RandomForestClassifier()
clf1.fit(X_Train, y_train)


with open("model_pickle", "wb") as file:
    pickle.dump(clf1, file)
    print("Done model")
