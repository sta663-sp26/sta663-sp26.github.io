import numpy as np
import pandas as pd

## Exercise 1

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
                     int(d.shape[0]/2)
  )
  .pivot(
    index=["country","year"], 
    columns="type", 
    values="rate"
  )
  .reset_index()
)