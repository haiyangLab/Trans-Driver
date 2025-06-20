import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

# 1. True positive gene set
train = pd.read_csv('../data/PCAWG.csv')
true_genes = set(train['gene'])

# 2. Driver gene set
drivers = set(pd.read_csv('../data/driver_gene/Trans-Driver_driver_gene.csv')['gene'])

# 3. Contingency table (2x2)
a = len(drivers & true_genes)           # Driver gene and true positive
b = len(drivers - true_genes)           # Driver gene but not true positive
c = len(true_genes - drivers)           # True positive but not driver gene

d = 20000 - a - b - c                   # Neither driver gene nor true positive
if d < 0:
    d = 0  # Avoid negative values
table = [[a, b], [c, d]]

oddsratio, pvalue = fisher_exact(table, alternative='greater')

print(f'Contingency table: {table}')
print(f'Fisher\'s exact test p-value: {pvalue:.4g}, -log10(P): {(-np.log10(pvalue)):.2f}')
