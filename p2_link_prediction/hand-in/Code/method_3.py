from utilities import np, nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

class GCN(nn.Module):
    def __init__(self, g, node_to_row, hidden_dim, out_dim):
        super().__init__()
        A = nx.to_numpy_array(g, dtype=np.float32)
        A_hat = A + np.eye(A.shape[0], dtype=np.float32)    # adjacency matrix with self-loops

        deg = A_hat.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        self.A_norm = torch.from_numpy(D_inv_sqrt @ A_hat @ D_inv_sqrt).float().to(device)

        X = (deg - deg.mean()) / (deg.std() + 1e-6)
        X = X.reshape(-1, 1)  # [N, 1]
        self.X = torch.from_numpy(X).float().to(device)

        self.node_to_row = node_to_row
        self.W1 = nn.Linear(1, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, candidates):
        # layer 1 (input to hidden-dimensional)
        h = self.A_norm @ self.W1(self.X)
        h = F.relu(h)
        h = self.dropout(h)

        # layer 2 (hidden-dimensional to embedding-dimensional)
        emb = self.A_norm @ self.W2(h)

        u = torch.tensor([self.node_to_row[int(c[0])] for c in candidates], dtype=torch.long)
        v = torch.tensor([self.node_to_row[int(c[1])] for c in candidates], dtype=torch.long)

        logits = (emb[u] * emb[v]).sum(dim=1)  # dot product for link prediction
        return logits


def method_3(g, node_to_row, candidates):
    HID_SIZE = 64
    EMB_SIZE = 12

    model = GCN(g, node_to_row, hidden_dim=HID_SIZE, out_dim=EMB_SIZE)
    model.load_state_dict(torch.load("method_3.pt"))

    with torch.no_grad():
        model.eval()
        scores = {
            tuple(edge): float(score)
            for edge, score in zip(candidates, model(torch.tensor(candidates)).cpu().numpy().reshape(-1))
        }
        
    return scores