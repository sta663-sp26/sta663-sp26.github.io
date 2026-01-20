## Exercise 1

x = {"a": 1, "b": 2, "c": 3}
y = {"c": 5, "d": 6, "e": 7}

def merge(x: dict, y: dict) -> dict:
    z = x.copy()
    for k,v in y.items():
        z[k] = v
    return z

print(merge(x,y), "\n", x, "\n", y)


def merge(x: dict, y: dict) -> dict:
    z = x.copy()
    z.update(y)
    return z

print(merge(x,y), "\n", x, "\n", y)


def merge(x: dict, y: dict) -> dict:
    return {**x, **y}

print(merge(x,y), "\n", x, "\n", y)

## Exercise 2

# A fixed collection of 100 integers.
# - A vector / array

# A queue (first in first out) of customer records.
# - A deque

# A stack (first in last out) of customer records.
# - A vector / array

# A count of word occurrences within a document. 
# - A dictionary mapping words to their counts.

# The heights of the bars in a histogram with even binwidths
# - A vector / array
