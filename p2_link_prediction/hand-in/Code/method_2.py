from utilities import np, nx
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cosine

def embed(g, node_to_row, candidates):
    EMB_SIZE = 10

    L = nx.laplacian_matrix(g)
    vals, vecs = eigsh(L, k=EMB_SIZE+1, which="SM")  # smallest K+1 smallest eigenvalues and their eigenvectors
    M = vecs[:, 1:]  # skipping trivial first eigenvector

    # lookup required: M rows are ordered like g_full.nodes() so not in node ID order
    return {
        tuple(candidate): (M[node_to_row[candidate[0]]], M[node_to_row[candidate[1]]])
        for candidate in candidates
    }

def method_2(g, node_to_row, candidates):
    scores = {}
    embeddings = embed(g, node_to_row, candidates)
    for candidate, (u_embed, v_embed) in embeddings.items():  # both positive and negative edges
        scores[candidate] = 1 - cosine(u_embed, v_embed)
    return scores