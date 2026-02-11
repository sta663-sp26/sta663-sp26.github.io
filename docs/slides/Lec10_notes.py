import numpy as np
import pandas as pd

## Exercise 1

df = pd.DataFrame({
  "country": ["A","A","B","B","C","C"],
  "year":    [1999, 2000, 1999, 2000, 1999, 2000],
  "rate":    ["0.7K/19M", "2K/20M", "37K/172M", "80K/174M", "212K/1T", "213K/1T"]
})

# Opt 1

df.assign(
  counts = lambda d: d.rate.str.split("/").str[0],
  pop    = lambda d: d.rate.str.split("/").str[1]
)

df.assign(
  rate = lambda d: d.rate.str.split("/"),
  counts = lambda d: d.rate.str[0],
  pop    = lambda d: d.rate.str[1]
).drop("rate", axis=1)

# Opt 2

( df.assign(
    rate = lambda d: d.rate.str.split("/")
  )
  .explode("rate")
  .assign(
    type = lambda d: ["cases", "pop"] * 
                     int(len(d)/2)
  )
  .pivot(
    index=["country","year"], 
    columns="type", 
    values="rate"
  )
  .reset_index()
)
