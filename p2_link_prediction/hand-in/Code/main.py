from utilities import load_network, display_accuracy, display_test_results, split_dataset
from method_2 import method_2

def main():
    # Load the data
    edges, g_full, non_edges, node_to_row, edges_test = load_network("../Data/trainingset.csv", "../Data/testset.csv")

    # Split the data
    g_baseline, X_train, Y_train, X_val, Y_val = split_dataset(edges, g_full, non_edges)
 
    # Accuracy on validation set for method 1
    scores = method_2(g_full, node_to_row, X_val)
    display_accuracy(scores, Y=Y_val)

    # Predictions on test set for method 1
    scores = method_2(g_full, node_to_row, edges_test)
    display_test_results(scores, top_k=100)

if __name__ == "__main__":
    main()