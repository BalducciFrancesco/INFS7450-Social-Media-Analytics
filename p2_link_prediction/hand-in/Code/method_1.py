from utilities import np, nx

def method_1(g_full, edges_test):
    scores = {}
    for candidate in edges_test:  # both positive and negative edges
        u, v = candidate
        scores[(u, v)] = len(list(nx.common_neighbors(g_full, u, v)))  # in the graph with only positive edges
    return scores