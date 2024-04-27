import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

df = pd.DataFrame(
    [[1, 2, 0], [3, 4, 1], [5, 6, 0], [7, 8, 1]],
    columns=["num", "amount", "target"]
)

# regression model
reg = LinearRegression().fit(df[["num", "amount"]], df["target"])
reg.score(df[["num", "amount"]], df["target"])

#classification model
clf = LogisticRegression().fit(df[["num", "amount"]], df["target"])
clf.score(df[["num", "amount"]], df["target"])