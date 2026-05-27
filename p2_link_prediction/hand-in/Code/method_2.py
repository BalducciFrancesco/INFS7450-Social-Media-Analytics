from utilities import np, nx
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cosine

def embed_candidates(g_full, node_to_row, edges_test):
    EMB_SIZE = 10

    L = nx.laplacian_matrix(g_full)
    vals, vecs = eigsh(L, k=EMB_SIZE+1, which="SM")  # smallest K+1 smallest eigenvalues and their eigenvectors
    M = vecs[:, 1:]  # skipping trivial first eigenvector

    # lookup required: M rows are ordered like g_full.nodes() so not in node ID order
    embed_candidate = lambda candidate: (M[node_to_row[candidate[0]]], M[node_to_row[candidate[1]]])
    return {candidate: embed_candidate(candidate) for candidate in edges_test}

def method_2(g_full, node_to_row, edges_test):
    scores = {}
    embeddings = embed_candidates(g_full, node_to_row, edges_test)
    for candidate, (u_embed, v_embed) in embeddings.items():  # both positive and negative edges
        scores[candidate] = 1 - cosine(u_embed, v_embed)
    return scores