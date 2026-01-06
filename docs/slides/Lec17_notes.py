import torch

## Exercise 1

a = torch.rand(4, 3, 2)
b = torch.rand(3, 2)
c = torch.rand(2, 3)
d = torch.rand(0) 
e = torch.rand(3, 1)
f = torch.rand(1, 2)

a*b # Yes
a*c # No
a*d # No
a*e # Yes
a*f # Yes

b*c # No
b*d # No
b*e # Yes
b*f # Yes

c*d # No
c*e # No
c*f # No

d*e # Yes
d*f # No

e*f # Yes


## Exercise 2

a = torch.ones(4,3,2)
b = torch.rand(3)
c = torch.rand(5,3)

a * b.unsqueeze(1)
a.unsqueeze(1) * c.unsqueeze(2)
b * c
