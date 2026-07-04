#!/usr/bin/env python3
import os, glob, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def safe_tau_from_filename(fn):
    nums = re.findall(r'[\d\.]+', fn)
    if not nums:
        return None
    s = nums[-1].strip('.')  # ← 关键修复
    try:
        return float(s)
    except:
        return None

# ---------------------------------
# main
# ---------------------------------
root = "."
purity_csv = os.path.join(root, "wn18rr_purity_summary.csv")
outdir = os.path.join(root, "plots")
os.makedirs(outdir, exist_ok=True)

pur = pd.read_csv(purity_csv)
pur = pur.sort_values("tau", ascending=False)

# ---- plot purity curve ----
plt.figure(figsize=(6,4))
plt.plot(pur['tau'], pur['avg_purity'], marker='o')
plt.gca().invert_xaxis()
plt.xlabel("τ")
plt.ylabel("Average Lexical-Domain Purity")
plt.title("WN18RR: Purity vs τ")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "avg_purity_vs_tau.png"), dpi=200)
plt.close()

# ---- modularity ----
mods = []
for fn in glob.glob(os.path.join(root, "louvain_tau_*.csv")):
    t = safe_tau_from_filename(fn)
    if t is None: continue
    df = pd.read_csv(fn)
    mods.append([t, "louvain", len(df), df.get('size', pd.Series()).mean()])

for fn in glob.glob(os.path.join(root, "leiden_tau_*.csv")):
    t = safe_tau_from_filename(fn)
    if t is None: continue
    df = pd.read_csv(fn)
    mods.append([t, "leiden", len(df), df.get('size', pd.Series()).mean()])

if mods:
    moddf = pd.DataFrame(mods, columns=["tau","method","n_communities","avg_comm_size"])
    merged = pur.merge(
        moddf.groupby('tau').agg({'n_communities':'sum','avg_comm_size':'mean'}).reset_index(),
        on='tau',how='left'
    )
    merged.to_csv(os.path.join(outdir,"purity_modularity_merged.csv"), index=False)

    plt.figure(figsize=(6,4))
    plt.scatter(merged['n_communities'], merged['avg_purity'])
    for i,row in merged.iterrows():
        plt.annotate(f"{row['tau']:.2f}", (row['n_communities'], row['avg_purity']))
    plt.xlabel("#Communities")
    plt.ylabel("Avg Purity")
    plt.title("Modularity vs Purity")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"modularity_vs_purity.png"), dpi=200)
    plt.close()
else:
    print("⚠ No louvain/leiden csv found → skip modularity plot.")
