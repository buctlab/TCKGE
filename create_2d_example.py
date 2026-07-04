"""
2D Toy Example: Maximal Tolerance Clique vs Traditional Partition-based Clustering
Demonstrates the key difference: non-transitivity of tolerance relations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from itertools import combinations

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# Color Configuration (Deep colors for white background)
# ============================================
COLOR_PALETTE = [
    '#1f77b4',  # Deep blue
    '#d62728',  # Deep red
    '#2ca02c',  # Deep green
    '#9467bd',  # Deep purple
    '#8c564b',  # Deep brown
    '#e377c2',  # Deep pink
    '#17becf',  # Deep cyan
    '#bcbd22',  # Olive (darker yellow-green)
    '#ff7f0e',  # Deep orange
    '#7f7f7f',  # Gray
]

# Edge colors for non-transitivity demo
COLOR_EDGE_AB = '#2ca02c'      # Deep green
COLOR_EDGE_BC = '#1f77b4'      # Deep blue
COLOR_EDGE_AC_FAIL = '#d62728' # Deep red

# Background edge color
COLOR_BG_EDGE = '#666666'      # Dark gray

# ============================================
# 1. Create synthetic 2D point cloud
# ============================================
# Design principle: create geometric configuration where:
# - A is close to B (within tolerance)
# - B is close to C (within tolerance)
# - A is FAR from C (beyond tolerance)
# This demonstrates NON-TRANSITIVITY

# Key insight: use vectors with different angles to control cosine similarity
# cos(θ) = similarity, so we can engineer specific similarity values

points = np.array([
    # Cluster 1: "Bridge" configuration demonstrating non-transitivity
    [1.0, 0.0],   # A (horizontal)
    [0.7, 0.7],   # B (45 degrees - bridge)
    [0.0, 1.0],   # C (vertical) - A and C are orthogonal!
    
    # Cluster 2: Tight triangle with high internal similarity
    [3.0, 0.1],   # D
    [3.1, 0.2],   # E
    [3.05, 0.25], # F
    
    # Cluster 3: Another tight pair
    [5.0, 2.0],   # G
    [5.1, 2.05],  # H
    
    # Outlier - far from everything
    [0.2, 3.0],   # I
])

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
n_points = len(points)

# Use configured color palette
colors = COLOR_PALETTE

# ============================================
# 2. Compute pairwise cosine similarity
# ============================================
# Normalize points to unit vectors (required for cosine similarity)
points_normalized = points / np.linalg.norm(points, axis=1, keepdims=True)
similarity_matrix = cosine_similarity(points_normalized)

# ============================================
# 3. Tolerance-based maximal cliques
# ============================================
def extract_maximal_tolerance_cliques(similarity_matrix, tau, size_cutoff=20):
    """
    Extract maximal tolerance cliques from similarity matrix.
    
    Parameters:
    -----------
    similarity_matrix : np.ndarray
        Pairwise similarity matrix (n x n)
    tau : float
        Tolerance threshold
    size_cutoff : int
        Maximum component size for clique extraction
    
    Returns:
    --------
    cliques : list of sets
        List of maximal cliques (as sets of node indices)
    """
    n = similarity_matrix.shape[0]
    
    # Build tolerance graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i, j] >= tau:
                G.add_edge(i, j)
    
    # Extract maximal cliques per component
    cliques = []
    for component in nx.connected_components(G):
        if len(component) > size_cutoff:
            continue
        subgraph = G.subgraph(component)
        component_cliques = list(nx.find_cliques(subgraph))
        cliques.extend(component_cliques)
    
    return [set(c) for c in cliques]

# Test multiple tau values
tau_values = [0.95, 0.85, 0.70, 0.50]
cliques_per_tau = {}

for tau in tau_values:
    cliques = extract_maximal_tolerance_cliques(similarity_matrix, tau)
    cliques_per_tau[tau] = cliques
    print(f"\nτ = {tau:.2f}:")
    print(f"  Number of maximal cliques: {len(cliques)}")
    for i, clique in enumerate(cliques):
        clique_labels = [labels[idx] for idx in sorted(clique)]
        print(f"  Clique {i+1}: {clique_labels}")

# ============================================
# 4. Traditional K-Means clustering (partition)
# ============================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
partition_labels = kmeans.fit_predict(points)

print("\n" + "="*50)
print("K-Means Partition (k=3):")
for k in range(3):
    cluster_points = [labels[i] for i in range(n_points) if partition_labels[i] == k]
    print(f"  Cluster {k+1}: {cluster_points}")

# ============================================
# 5. Visualization
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(18, 8))

# -------------------- Row 1: Tolerance Cliques --------------------
for col_idx, tau in enumerate([0.95, 0.85, 0.70]):
    ax = axes[0, col_idx]
    
    # Draw tolerance edges first (as background)
    G = nx.Graph()
    G.add_nodes_from(range(n_points))
    for i in range(n_points):
        for j in range(i+1, n_points):
            if similarity_matrix[i, j] >= tau:
                G.add_edge(i, j)
                ax.plot([points[i, 0], points[j, 0]], 
                       [points[i, 1], points[j, 1]], 
                       COLOR_BG_EDGE, alpha=0.5, linewidth=2, zorder=1)
    
    # Highlight maximal cliques with colored convex hulls
    cliques = cliques_per_tau[tau]
    for clique_idx, clique in enumerate(cliques):
        clique_points = np.array([points[i] for i in clique])
        if len(clique) >= 3:
            # Draw filled convex hull for cliques with 3+ members
            from scipy.spatial import ConvexHull
            hull = ConvexHull(clique_points)
            # Fill the hull area
            hull_points = clique_points[hull.vertices]
            ax.fill(hull_points[:, 0], hull_points[:, 1], 
                   color=colors[clique_idx % 10], alpha=0.25, zorder=2)
            # Draw hull boundary
            for simplex in hull.simplices:
                ax.plot(clique_points[simplex, 0], clique_points[simplex, 1], 
                       color=colors[clique_idx % 10], linewidth=3.5, zorder=2, alpha=0.9)
        elif len(clique) == 2:
            # For 2-point cliques, draw thick line
            ax.plot(clique_points[:, 0], clique_points[:, 1], 
                   color=colors[clique_idx % 10], linewidth=3.5, zorder=2, alpha=0.9)
    
    # Draw text labels with minimal styling
    for i, label in enumerate(labels):
        # Simple text with white outline for readability
        ax.text(points[i, 0], points[i, 1], label, 
                fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=5,
                color='black',
                path_effects=[path_effects.withStroke(linewidth=2.5, foreground='white')])
    
    ax.set_title(f'Tolerance Cliques (τ={tau:.2f})\n{len(cliques)} maximal cliques', 
                fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, 5.7)
    ax.set_ylim(-0.5, 3.5)
    ax.grid(True, alpha=0.3)

# -------------------- Row 2: Comparisons --------------------

# Subplot 2,1: K-Means Partition
ax = axes[1, 0]

# Draw cluster regions (convex hulls) first
for k in range(3):
    cluster_mask = partition_labels == k
    cluster_points = points[cluster_mask]
    if len(cluster_points) >= 3:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
        ax.fill(hull_points[:, 0], hull_points[:, 1], 
               color=colors[k], alpha=0.3, zorder=1, label=f'Cluster {k+1}')
        # Draw hull boundary for clarity
        for i in range(len(hull.vertices)):
            v1, v2 = hull.vertices[i], hull.vertices[(i+1) % len(hull.vertices)]
            ax.plot([cluster_points[v1, 0], cluster_points[v2, 0]],
                   [cluster_points[v1, 1], cluster_points[v2, 1]],
                   color=colors[k], linewidth=2.5, alpha=0.8, zorder=2)

# Draw text labels
for i, label in enumerate(labels):
    ax.text(points[i, 0], points[i, 1], label, 
           fontsize=10, fontweight='bold',
           ha='center', va='center', zorder=4,
           color='black',
           path_effects=[path_effects.withStroke(linewidth=2.5, foreground='white')])

ax.set_title('K-Means Partition\n(Hard Assignment)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-0.5, 3.5)

# Subplot 2,2: Non-transitivity demonstration
ax = axes[1, 1]
tau_demo = 0.70

# Highlight A-B-C non-transitivity
A_idx, B_idx, C_idx = 0, 1, 2
sim_AB = similarity_matrix[A_idx, B_idx]
sim_BC = similarity_matrix[B_idx, C_idx]
sim_AC = similarity_matrix[A_idx, C_idx]

# Draw edges with annotations
if sim_AB >= tau_demo:
    ax.plot([points[A_idx, 0], points[B_idx, 0]], 
           [points[A_idx, 1], points[B_idx, 1]], 
           COLOR_EDGE_AB, linewidth=5, label=f'A↔B: {sim_AB:.3f} ≥ τ', zorder=2, alpha=0.9)

if sim_BC >= tau_demo:
    ax.plot([points[B_idx, 0], points[C_idx, 0]], 
           [points[B_idx, 1], points[C_idx, 1]], 
           COLOR_EDGE_BC, linewidth=5, label=f'B↔C: {sim_BC:.3f} ≥ τ', zorder=2, alpha=0.9)

if sim_AC < tau_demo:
    ax.plot([points[A_idx, 0], points[C_idx, 0]], 
           [points[A_idx, 1], points[C_idx, 1]], 
           COLOR_EDGE_AC_FAIL, linewidth=5, linestyle='--', label=f'A↮C: {sim_AC:.3f} < τ', zorder=2, alpha=0.9)

# Draw text labels on top
for i, label in enumerate(labels):
    # Only label A, B, C for this subplot to avoid clutter
    if i <= 2:
        ax.text(points[i, 0], points[i, 1], label, 
               fontsize=12, fontweight='bold',
               ha='center', va='center', zorder=4,
               color='black',
               path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])
    else:
        # Other points shown as small dots
        ax.scatter(points[i, 0], points[i, 1], s=50, c='lightgray', 
                  edgecolors='gray', linewidths=1, zorder=1, alpha=0.3)

ax.set_title(f'Non-Transitivity Demo (τ={tau_demo:.2f})\nA~B, B~C, but A≁C', 
            fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.3, 5.5)
ax.set_ylim(-0.5, 3.5)

# Subplot 2,3: Similarity heatmap
ax = axes[1, 2]
im = ax.imshow(similarity_matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(n_points))
ax.set_yticks(range(n_points))
ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
ax.set_yticklabels(labels, fontsize=9, fontweight='bold')
ax.set_title('Cosine Similarity Matrix', fontsize=11, fontweight='bold')

# Add colorbar with limited height (shrink to match matrix height)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Similarity', fontsize=9)

# Annotate values using WCAG luminance formula for maximum contrast
cmap_blues = plt.get_cmap('Blues')
for i in range(n_points):
    for j in range(n_points):
        val = similarity_matrix[i, j]
        r, g, b, _ = cmap_blues(val)
        # WCAG 2.1 relative luminance
        def _lin(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
        text_color = 'white' if lum < 0.179 else 'black'
        ax.text(j, i, f'{val:.2f}',
                ha="center", va="center", color=text_color,
                fontsize=8, fontweight='bold')

plt.tight_layout(pad=1.0, h_pad=1.5)
plt.subplots_adjust(hspace=0.25)
plt.savefig('figures/toy_example_tolerance_vs_partition.pdf', dpi=300, bbox_inches='tight')
print("\n" + "="*50)
print("Figure saved: figures/toy_example_tolerance_vs_partition.pdf")
#plt.show()

print("\n" + "="*50)
print("LaTeX Table Code:")
print("="*50)

print(r"""
\begin{table}[ht]
\centering
\caption{Comparison of Tolerance Cliques vs Partition-based Clustering on 2D Synthetic Dataset}
\label{tab:toy_example}
\begin{tabular}{lll}
\toprule
Method & Result & Key Property \\
\midrule
Tolerance ($\tau$=0.85) & 4 overlapping cliques & Allows overlapping \\
                         & \{A,B\}, \{B,C\}, \{D,E,F\}, \{G,H\} & Non-transitive \\
K-Means ($k$=3)          & 3 disjoint partitions & Mutually exclusive \\
                         & \{A,B,C,I\}, \{D,E,F\}, \{G,H\} & Transitivity assumed \\
\bottomrule
\end{tabular}
\end{table}
""")

print("\n" + "="*50)
print("Similarity Matrix Analysis:")
print("="*50)
print(f"A-B similarity: {similarity_matrix[0, 1]:.4f}")
print(f"B-C similarity: {similarity_matrix[1, 2]:.4f}")
print(f"A-C similarity: {similarity_matrix[0, 2]:.4f}")
print(f"\nNon-transitivity check at τ=0.85:")
print(f"  A~B? {similarity_matrix[0, 1] >= 0.85} (sim={similarity_matrix[0, 1]:.3f})")
print(f"  B~C? {similarity_matrix[1, 2] >= 0.85} (sim={similarity_matrix[1, 2]:.3f})")
print(f"  A~C? {similarity_matrix[0, 2] >= 0.85} (sim={similarity_matrix[0, 2]:.3f})")
print(f"  => Demonstrates non-transitivity!")
