## Exercise 1

# Create a list containing tuples of x and y coordinates of all points of a 
# regular grid for x∈[0,10] and y∈[0,10]

g = [(x,y) for x in range(11) for y in range(11)]
print(len(g))

# Count the number of points where y>x

len([(x,y) for x in range(11) for y in range(11) if y > x])
len([(x,y) for x,y in g if y > x])
     
# Count the number of points x or y is prime.

prime = (2,3,5,7)
len([(x,y) for x in range(11) for y in range(11) if x in prime or y in prime])
len([(x,y) for x,y in g if x in prime or y in prime])


## Exercise 2

# Write a function, kg_to_lb, that converts a list of weights in kilograms to 
# a list of weights in pounds (there a 1 kg = 2.20462 lbs). 
# Include a doc string and function annotations.

def kg_to_lb(wt: list) -> list:
    """Convert weights in kilograms to pounds"""
    
    return [x * 2.20462 for x in wt]

kg_to_lb([1,2,5,3])
     
# Write a second function, total_lb, that calculates the total weight in pounds 
# of an order, the input arguments should be a list of item weights in kilograms 
# and a list of the number of each item ordered.

def total_lb(wt: list, n: list) -> list:
    """Calculate the total weight of an order (in pounds)
    given the item weights (in kgs) and the number ordered
    """
    
    return sum([x * y * 2.20462 for x,y in zip(wt, n)])

total_lb([1,2,5,3], [1,2,2,1])

