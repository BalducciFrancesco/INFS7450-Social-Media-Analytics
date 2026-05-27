from utilities import np, nx

def method_1(g, candidates):
    scores = {}
    for candidate in candidates:  # both positive and negative edges
        u, v = candidate
        scores[(u, v)] = len(list(nx.common_neighbors(g, u, v)))  # in the graph with only positive edges
    return scores