import json
import matplotlib.pyplot as plt
import numpy as np

sizes = []
with open('cliques_tau_0.450.json','r') as f:
    cliques = json.load(f)
    for c in cliques:
        sizes.append(len(c))

plt.figure(figsize=(6,4))
plt.hist(np.log10(sizes), bins=30, alpha=0.75)
plt.xlabel('log10(cluster size)')
plt.ylabel('frequency')
plt.title('Tolerance Clique log-size distribution (τ=0.45)')
plt.tight_layout()
plt.savefig('clique_log_distribution.pdf')
plt.close()