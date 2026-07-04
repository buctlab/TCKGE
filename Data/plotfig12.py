import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("purity_modularity_merged.csv")

x = df['avg_purity']
y = df['n_communities']
size = df['avg_comm_size'] / 10  # scale

plt.figure(figsize=(6,4))
plt.scatter(x, y, s=size, c=df['tau'], cmap='viridis')
for i,row in df.iterrows():
    plt.text(row['avg_purity'], row['n_communities'], f"{row['tau']:.2f}")

plt.xlabel('Average Lexical-Domain Purity')
plt.ylabel('Number of Communities')
plt.title('Modularity vs Purity Tradeoff')
plt.tight_layout()
plt.savefig('modularity_tradeoff.pdf')
plt.close()
