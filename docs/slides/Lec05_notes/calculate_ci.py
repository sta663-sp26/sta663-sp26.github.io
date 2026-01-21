import numpy as np
from scipy import stats

data = [23, 25, 28, 30, 26, 27, 29, 24, 31, 28]

mean = np.mean(data)
sem = stats.sem(data)
ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)

print(f"Sample mean: {mean:.2f}")
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
