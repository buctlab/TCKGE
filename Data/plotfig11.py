import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("summary_robustness.csv")

plt.figure(figsize=(6,4))
plt.plot(df['tau'], df['n_louvain_communities'], marker='o', label='Louvain')
plt.plot(df['tau'], df['n_leiden_communities'], marker='s', label='Leiden')
plt.xlabel(r'$\tau$')
plt.ylabel('Number of communities')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('louvain_vs_leiden.pdf')
plt.close()
