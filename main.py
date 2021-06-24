from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
data = pd.read_csv("Train.csv")
data.head()

# visualising the data
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

plt.show()