# wn18rr_full_analysis.py
import numpy as np, json, os, argparse, time, gc
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import community as community_louvain   # python-louvain
import igraph as ig
import leidenalg
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

def load_embeddings(embed_path):
    E = np.load(embed_path)
    # normalize rows for cosine
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / (norms + 1e-12)
    return E

def build_sparse_topk(E, k):
    # Use sklearn NearestNeighbors (cosine) to find top-k for each node
    n = E.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='auto').fit(E)
    dist, idx = nbrs.kneighbors(E)  # idx includes self at pos 0
    rows, cols = [], []
    for i in range(n):
        for j in idx[i,1:]:
            rows.append(i); cols.append(j)
    # symmetrize (OR)
    rows2 = rows + cols
    cols2 = cols + rows
    data = np.ones(len(rows2), dtype=np.int8)
    A = coo_matrix((data, (rows2, cols2)), shape=(n,n)).tocsr()
    return A

def build_sparse_threshold_from_topk(E, k=200):
    # alternative: compute top-200 similarities only (approx) and apply percentile thresholds later
    # We'll get cosine values for top-k neighbors for each node.
    n = E.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(E)
    dist, idx = nbrs.kneighbors(E)  # dist is cosine distance
    sim = 1 - dist  # convert to cosine similarity
    # collect edges and similarity values
    rows, cols, vals = [], [], []
    for i in range(n):
        for pos, j in enumerate(idx[i,1:]):
            rows.append(i); cols.append(j); vals.append(float(sim[i, pos+1-1]))
    # make symmetric by taking max(sim_ij, sim_ji)
    df = pd.DataFrame({'i':rows,'j':cols,'s':vals})
    df2 = df.groupby(['i','j'], as_index=False)['s'].max()
    # make symmetric by union i->j and j->i: we'll add reversed edges
    df_rev = df2.rename(columns={'i':'j','j':'i','s':'s2'})[['i','j','s2']]
    merged = pd.merge(df2, df_rev, on=['i','j'], how='outer')
    merged['s'] = merged[['s','s2']].max(axis=1)
    merged = merged[['i','j','s']].dropna()
    rows = merged['i'].astype(int).tolist()
    cols = merged['j'].astype(int).tolist()
    vals = merged['s'].astype(float).tolist()
    A = coo_matrix((vals, (rows, cols)), shape=(n,n))
    # symmetrize numeric matrix
    A = A + A.T
    # convert to csr
    return A.tocsr()

def threshold_sparse_from_edge_list(A_weighted, tau):
    # A_weighted: csr with edge weights (similarity)
    A_bin = A_weighted.copy()
    A_bin.data = (A_bin.data >= tau).astype(np.int8)
    A_bin.eliminate_zeros()
    return A_bin

def analyze_components(A_bin):
    from scipy.sparse.csgraph import connected_components
    n_comp, labels = connected_components(csgraph=A_bin, directed=False, return_labels=True)
    sizes = np.bincount(labels)
    return {
        "n_components": int(n_comp),
        "component_sizes": sizes.tolist(),
        "largest": int(sizes.max())
    }


def extract_maximal_cliques_in_components(A_bin, idmap_inv, size_cutoff_for_bk=120, max_cliques_per_comp=2000):
    # Convert sparse matrix to networkx graph (NetworkX 3.x API)
    G = nx.from_scipy_sparse_array(A_bin)

    cliques = []
    for comp in nx.connected_components(G):
        comp = list(comp)

        # ignore trivial components
        if len(comp) <= 1:
            continue

        # we avoid Bron–Kerbosch on large components
        if len(comp) > size_cutoff_for_bk:
            continue

        subG = G.subgraph(comp)

        # find maximal cliques (networkx.find_cliques is BK)
        found = list(nx.find_cliques(subG))

        # Limit number of cliques stored
        if len(found) > max_cliques_per_comp:
            found = found[:max_cliques_per_comp]

        for c in found:
            cliques.append([idmap_inv[v] for v in c])

    return cliques


def run_louvain(A_bin, idmap_inv):
    G = nx.from_scipy_sparse_array(A_bin)
    partition = community_louvain.best_partition(G)
    clusters = {}
    for node, com in partition.items():
        clusters.setdefault(com, []).append(idmap_inv[node])
    return clusters


def run_leiden(A_bin, idmap_inv, resolution_parameter=1.0):
    # convert to igraph
    sources, targets = A_bin.nonzero()
    weights = A_bin.data.tolist()
    g = ig.Graph(n=A_bin.shape[0], edges=list(zip(sources.tolist(), targets.tolist())), directed=False)
    if len(weights) == g.ecount():
        g.es['weight'] = weights
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution_parameter)
    clusters = {}
    for cid, memb in enumerate(partition):
        clusters[cid] = [idmap_inv[v] for v in memb]
    return clusters

def baseline_random_graph(A_bin, seed=42):
    # generate random graph with same degree sequence via configuration model (simple)
    degrees = np.array(A_bin.sum(axis=1)).ravel().astype(int)
    # use networkx degree_sequence_graph to generate simple graph
    try:
        G_rand = nx.configuration_model(degrees, seed=seed)
        G_rand = nx.Graph(G_rand)  # remove parallel edges
        G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
        return nx.to_numpy_array(G_rand)
    except Exception as e:
        print("random baseline fail:", e)
        return None

def run_experiments(embed_path, idmap_path, outdir, taus, ks, topk_for_similarity=200):
    os.makedirs(outdir, exist_ok=True)
    E = load_embeddings(embed_path)
    n = E.shape[0]
    with open(idmap_path,'r',encoding='utf-8') as f:
        idmap = json.load(f)
    # idmap maps entity -> idx (string ints maybe)
    idmap_inv = {int(v):k for k,v in idmap.items()}
    # Build a weighted sparse neighborhood (top topk_for_similarity neighbors) only once
    print("Building weighted neighbor graph top-k:", topk_for_similarity)
    A_weighted = build_sparse_threshold_from_topk(E, k=topk_for_similarity)  # weighted cos sim
    # compute global similarity percentiles if needed: collect weights
    weights = A_weighted.data
    # For each tau produce binarized graph and analyze
    summary_rows = []
    for tau in taus:
        print("Processing tau:", tau)
        A_bin = threshold_sparse_from_edge_list(A_weighted, tau)
        comp = analyze_components(A_bin)
        # run louvain/leiden
        try:
            louv = run_louvain(A_bin, idmap_inv)
            leiden = run_leiden(A_bin, idmap_inv)
        except Exception as e:
            print("community detection error:", e)
            louv = {}
            leiden = {}
        # extract cliques in small components
        cliques = extract_maximal_cliques_in_components(A_bin, idmap_inv, size_cutoff_for_bk=120)
        # baseline random graph (attempt)
        # convert A_bin to networkx and then degree sequence baseline
        G = nx.from_scipy_sparse_array(A_bin)
        deg = [d for _,d in G.degree()]
        try:
            G_rand = nx.configuration_model(deg, seed=123)
            G_rand = nx.Graph(G_rand); G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
            rand_components = nx.number_connected_components(G_rand)
        except Exception as e:
            rand_components = None
        summary = {
            'tau': tau,
            'n_nodes': n,
            'n_edges_bin': int(A_bin.nnz/2),
            'n_components': comp['n_components'],
            'largest_component': comp['largest'],
            'n_cliques_extracted': len(cliques),
            'n_louvain_communities': len(louv),
            'n_leiden_communities': len(leiden),
            'rand_components': rand_components
        }
        summary_rows.append(summary)
        # save some outputs
        with open(os.path.join(outdir, f"cliques_tau_{tau}.json"), 'w', encoding='utf-8') as f:
            json.dump(cliques, f, ensure_ascii=False, indent=2)
        # save community examples
        pd.DataFrame({'community_id': list(louv.keys()), 'size':[len(v) for v in louv.values()]}).to_csv(os.path.join(outdir, f"louvain_tau_{tau}_summary.csv"), index=False)
        pd.DataFrame({'community_id': list(leiden.keys()), 'size':[len(v) for v in leiden.values()]}).to_csv(os.path.join(outdir, f"leiden_tau_{tau}_summary.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(outdir, "summary_robustness.csv"), index=False)
    print("Finished. outputs in", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", required=True)
    parser.add_argument("--idmap", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--taus", nargs="+", type=float, default=[0.95,0.9,0.85,0.8,0.75])
    parser.add_argument("--ks", nargs="+", type=int, default=[5,10,20])
    args = parser.parse_args()
    run_experiments(args.embed, args.idmap, args.out, taus=args.taus, ks=args.ks)
