
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import os

sns.set_style("whitegrid")

plt.rcParams['font.size'] = 10

plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = "./figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

transe_summary = pd.read_csv("./Data/summary_robustness.csv")

transe_summary["model"] = "transe"

rotate_summary = pd.read_csv("./Data/rotate_results/summary_robustness_rotate.csv")

try:

    transe_purity = pd.read_csv("./Data/transe_results/purity_summary_transe.csv")

except FileNotFoundError:

    transe_purity = pd.DataFrame({

        'tau': [0.45, 0.50, 0.55, 0.60, 0.65],

        'avg_purity': [0.909, 0.933, 0.909, 0.899, 0.938],

        'perfect_purity_ratio': [0.818, 0.865, 0.818, 0.797, 0.875]

    })

rotate_purity = pd.read_csv("./Data/rotate_results/purity_summary_rotate.csv")

print("=== begin ===")

fig, ax = plt.subplots(figsize=(10, 6))

transe_tau = transe_purity['tau']

ax.plot(transe_tau, transe_purity['avg_purity'], 

        marker='o', linewidth=2.5, markersize=10, color='#2E86AB', 

    label='TransE (Avg. purity)', linestyle='-')

ax.plot(transe_tau, transe_purity['perfect_purity_ratio'], 

        marker='s', linewidth=2, markersize=8, color='#2E86AB', 

    label='TransE (Perfect ratio)', linestyle='--', alpha=0.7)

rotate_tau = rotate_purity['tau']

ax.plot(rotate_tau, rotate_purity['avg_purity'], 

        marker='o', linewidth=2.5, markersize=10, color='#C73E1D', 

    label='RotatE (Avg. purity)', linestyle='-')

ax.plot(rotate_tau, rotate_purity['perfect_purity_ratio'], 

        marker='s', linewidth=2, markersize=8, color='#C73E1D', 

    label='RotatE (Perfect ratio)', linestyle='--', alpha=0.7)

ax.set_xlabel('Tolerance Threshold (τ)', fontsize=13)

ax.set_ylabel('Lexical-domain score', fontsize=13)

ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

ax.grid(True, alpha=0.3)

ax.invert_xaxis()

ax.set_ylim([-0.05, 1.05])

common_taus = sorted(set(transe_summary['tau']) & set(rotate_summary['tau']), reverse=True)

if common_taus:

    transe_cliques = [transe_summary[transe_summary['tau']==t]['n_cliques_extracted'].values[0] 

                      for t in common_taus]

    rotate_cliques = [rotate_summary[rotate_summary['tau']==t]['n_cliques_extracted'].values[0] 

                      for t in common_taus]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(common_taus))

    width = 0.35

    bars1 = ax.bar(x - width/2, transe_cliques, width, label='TransE', 

                   color='#2E86AB', alpha=0.8, edgecolor='black')

    bars2 = ax.bar(x + width/2, rotate_cliques, width, label='RotatE', 

                   color='#C73E1D', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Tolerance Threshold (τ)', fontsize=13)

    ax.set_ylabel('Number of Extracted Cliques', fontsize=13)

    ax.set_xticks(x)

    ax.set_xticklabels([f'{t:.2f}' for t in common_taus])

    ax.legend(loc='upper left', fontsize=11)

    ax.grid(True, alpha=0.3, axis='y')

    ax.set_yscale('log')

    for bars in [bars1, bars2]:

        for bar in bars:

            height = bar.get_height()

            if height > 0:

                ax.text(bar.get_x() + bar.get_width()/2., height,

                       f'{int(height)}',

                       ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/comparison_n_cliques.pdf", dpi=300, bbox_inches='tight')

    print(f"✓ Saved: comparison_n_cliques.pdf")

    plt.close()

if common_taus:

    transe_comps = [transe_summary[transe_summary['tau']==t]['n_components'].values[0] 

                    for t in common_taus]

    rotate_comps = [rotate_summary[rotate_summary['tau']==t]['n_components'].values[0] 

                    for t in common_taus]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(common_taus, transe_comps, marker='o', linewidth=2.5, 

            markersize=10, color='#2E86AB', label='TransE')

    ax.plot(common_taus, rotate_comps, marker='s', linewidth=2.5, 

            markersize=10, color='#C73E1D', label='RotatE')

    ax.set_xlabel('Tolerance Threshold (τ)', fontsize=13)

    ax.set_ylabel('Number of Components', fontsize=13)

    ax.set_title('Model Comparison: Graph Fragmentation', fontsize=15, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)

    ax.grid(True, alpha=0.3)

    ax.invert_xaxis()

if common_taus:

    transe_comps = [transe_summary[transe_summary['tau']==t]['n_components'].values[0]

                    for t in common_taus]

    rotate_comps = [rotate_summary[rotate_summary['tau']==t]['n_components'].values[0]

                    for t in common_taus]

    transe_largest = [transe_summary[transe_summary['tau']==t]['largest_component'].values[0]

                      for t in common_taus]

    rotate_largest = [rotate_summary[rotate_summary['tau']==t]['largest_component'].values[0]

                      for t in common_taus]

    fig, ax1 = plt.subplots(figsize=(4.2, 3.0))

    ax2 = ax1.twinx()

    line1, = ax1.plot(common_taus, transe_comps, marker='o', linewidth=1.8,

                      markersize=5, color='#2E86AB', label='TransE components')

    line2, = ax1.plot(common_taus, rotate_comps, marker='s', linewidth=1.8,

                      markersize=5, color='#C73E1D', label='RotatE components')

    line3, = ax2.plot(common_taus, transe_largest, marker='^', linewidth=1.6,

                      markersize=5, color='#2E86AB', linestyle='--', label='TransE largest')

    line4, = ax2.plot(common_taus, rotate_largest, marker='v', linewidth=1.6,

                      markersize=5, color='#C73E1D', linestyle='--', label='RotatE largest')

    ax1.set_xlabel('Tolerance Threshold (τ)', fontsize=8)

    ax1.set_ylabel('Number of components', fontsize=8)

    ax2.set_ylabel('Largest component (log scale)', fontsize=8)

    ax2.set_yscale('log')

    ax1.invert_xaxis()

    ax1.grid(True, alpha=0.3)

    ax1.tick_params(axis='both', labelsize=7)

    ax2.tick_params(axis='y', labelsize=7)

    lines = [line1, line2, line3, line4]

    ax1.legend(lines, [line.get_label() for line in lines], loc='upper center',

               bbox_to_anchor=(0.5, 1.22), fontsize=6.4, framealpha=0.9,

               ncol=2, columnspacing=0.9, handlelength=2.2)

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/comparison_connectivity.pdf", dpi=300, bbox_inches='tight')

    print(f"✓ Saved: comparison_connectivity.pdf")

    plt.close()

print("\n=== end ===")
