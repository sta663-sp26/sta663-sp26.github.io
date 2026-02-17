import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

## Exercise 1

books = pd.read_csv("https://sta663-sp26.github.io/slides/data/daag_books.csv")

X = pd.get_dummies(
  books[["volume", "cover"]]
).assign(
  vol_x_hb = X["volume"] * X["cover_hb"],
  vol_x_pb = X["volume"] * X["cover_pb"]
)

lm = LinearRegression(fit_intercept=False).fit(X=X, y=books.weight)

lm.coef_
lm.feature_names_in_

books["pred"] = lm.predict(X=X)

print( metrics.r2_score(books.weight, books.pred) )
print( metrics.root_mean_squared_error(books.weight, books.pred) )

plt.figure(layout="constrained")
sns.scatterplot(data=books, x="volume", y="weight", hue="cover")
sns.lineplot(data=books, x="volume", y="pred", hue="cover")
plt.show()
