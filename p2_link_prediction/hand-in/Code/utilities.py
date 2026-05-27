from sklearn.preprocessing import binarize
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


random.seed(42)

def load_network(train_csv_path, test_csv_path):
    """
    Args:
        train_csv_path (str): Path to the training CSV file containing edges.
        test_csv_path (str): Path to the testing CSV file containing edges.
    Returns:
        edges (list of tuples): List of edges in the training set.
        g_full (networkx.Graph): The full graph constructed from the training edges.
        non_edges (list of tuples): List of non-edges in the full graph.
        node_to_row (dict): Mapping from node IDs to row indices in the adjacency matrix.
        edges_test (list of tuples): List of edges in the testing set.
    """
    edges = np.loadtxt(train_csv_path, delimiter=",", dtype=int).tolist()
    edges_test = np.loadtxt(test_csv_path, delimiter=",", dtype=int).tolist()

    g_full = nx.Graph()
    g_full.add_edges_from(edges)
    non_edges = list(nx.non_edges(g_full))
    node_to_row = {node: i for i, node in enumerate(g_full.nodes())}

    return edges, g_full, non_edges, node_to_row, edges_test


def split_dataset(edges, g_full, non_edges):
    """
    Splits the dataset into training and testing sets.
    Args:
        edges (list of tuples): List of edges in the graph.
        g_full (networkx.Graph): The full graph constructed from the training edges.
        non_edges (list of tuples): List of non-edges in the full graph.
    Returns:
        g_baseline (networkx.Graph): A graph containing only positive edges.
        X_train (list of tuples): List of positive and negative edges in the training set.
        Y_train (list of int): List of labels for the training set (1 for edges, 0 for non-edges).
        X_val (list of tuples): List of positive and negative edges in the validation set.
        Y_val (list of int): List of labels for the validation set (1 for edges, 0 for non-edges).
    """
    pos_baseline, pos_temp = train_test_split(edges, test_size=0.5, random_state=42)
    pos_train, pos_val = train_test_split(pos_temp, test_size=0.4, random_state=42)

    g_baseline = nx.Graph()
    g_baseline.add_edges_from(pos_baseline)
    g_baseline.add_nodes_from(g_full.nodes())

    def add_negative_edges(X, Y):
        neg_edges = random.sample(non_edges, len(X))
        X = X + neg_edges
        Y = Y + [0] * len(neg_edges)
        return X, Y

    X_train, Y_train = pos_train, [1] * len(pos_train)
    X_train, Y_train = add_negative_edges(X_train, Y_train)

    X_val, Y_val = pos_val, [1] * len(pos_val)
    X_val, Y_val = add_negative_edges(X_val, Y_val)

    return g_baseline, X_train, Y_train, X_val, Y_val


def binarize_scores(scores, top_k):
    """
    Converts a free-range score list into binary predictions by selecting the top-K edges as positive.
    """
    idx = np.argsort(scores)[-top_k:]
    Y_pred = np.zeros(len(scores))
    Y_pred[idx] = 1
    return Y_pred


def display_accuracy(scores: dict[tuple[int, int], float], Y):
    """
    Converts a free-range score list into binary predictions and computes accuracy.
    Selecting the top-K edges as positive, where K is the number of true positive edges.
    """
    top_k = Y.count(1)
    Y_pred = binarize_scores(list(scores.values()), top_k)
    acc = accuracy_score(Y, Y_pred)
    print(f"Accuracy: {acc:.4f}")


def display_test_results(scores: dict[tuple[int, int], float], top_k):
    """Display top-K edges based on a link prediction scoring measure"""
    items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    print(f"Top {top_k} edges:")
    for pair, score in items:
        print(f"Edge {pair}: {score:.4f}")