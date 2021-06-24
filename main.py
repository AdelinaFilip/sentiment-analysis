import re
from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud


count = CountVectorizer()
data = pd.read_csv("Train.csv")
data.head()

# visualising the data


def data_distribution():
    fig = plt.figure(figsize=(6, 6))
    colors = ["skyblue", "pink"]

    pos = data[data["label"] == 1]
    neg = pos = data[data["label"] == 0]

    ck = [pos["label"].count(), neg["label"].count()]
    legpie = plt.pie(
        ck,
        labels=["Positive", "Negative"],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=(0, 0.1)
    )

    plt.savefig("data_distribution.png")


# remove HTML tags and emojis
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emojis).replace('-', '')
    return text


data['text'] = data['text'].apply(preprocessor)

porter = PorterStemmer()


# simplify the data
def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# remove the stopwords
nltk.download("stopwords")

stop = stopwords.words("english")

positive_data = data[data["label"] == 1]
positive_data = positive_data["text"]

negative_data = data[data["label"] == 0]
negative_data = negative_data["text"]


# draw wordcloud
def wordcloud_draw(data, color="white", filename=""):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                             if (word != 'movie' and word != 'film')
                             ])
    wordcloud = WordCloud(
        stopwords=stop,
        background_color=color,
        width=2500,
        height=2000
    ).generate(cleaned_word)

    plt.figure(1, figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(f"{filename}.png")


print("Positive words are as follows...")
wordcloud_draw(positive_data, 'white', "positive_words")

print("Negative words are as follows...")
wordcloud_draw(negative_data, 'white', "negative_words")


# confert the raw documents into a feature matrix
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None,
                        tokenizer=tokenizer_porter, use_idf=True, norm='l2', smooth_idf=True)
y = data.label.values
x = tfidf.fit_transform(data.text)

# split data into 50% training and 50% test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, random_state=1, test_size=0.5, shuffle=False)

# train
clf = LogisticRegressionCV(cv=6, scoring='accuracy', random_state=0,
                           n_jobs=-1, verbose=3, max_iter=500).fit(X_train, Y_train)

# predict
y_pred = clf.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(Y_test, y_pred) * 100, "%")