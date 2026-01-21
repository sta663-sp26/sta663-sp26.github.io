## Exercise 1 Solutions

# 1. Import `load_csv` from `loader.py`
from analytics.data.loader import load_csv

# 2. Import both functions from `cleaner.py`
from analytics.data.cleaner import remove_nulls, normalize

# 3. Import `LinearModel` with the alias `lm`
from analytics.models.regression import LinearModel as lm

# 4. From within `regression.py`, import `remove_nulls` using a relative import
#    Add to analytics/models/regression.py:
#
#    from ..data.cleaner import remove_nulls
#
#    Then verify by importing the module (the relative import runs when loaded):
#
#    import analytics.models.regression as reg
#    reg.remove_nulls

## Exercise 2 Solutions
#
# 1. Create a new directory and copy the script into it
#
#    mkdir ci_project
#    cd ci_project
#    cp /path/to/calculate_ci.py .
#
# 2. Initialize a uv project and add the required dependencies
#
#    uv init --bare
#    uv add numpy scipy
#
# 3. Run the script and verify the output
#
#    uv run python calculate_ci.py
#
#    Expected output:
#    Sample mean: 27.10
#    95% CI: (24.94, 29.26)