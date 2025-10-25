import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from data_loader import load_dataset
from utils.metrics import compute_classification_metrics, compute_regression_metrics
from models.gcn_standard import GCN
from models.gcn_regression import GCNNodeRegression
from models.gcn_from_scratch import GCNFromScratch


def get_model(name, cfg, dataset):
    """Factory to initialize selected GNN model."""
    if name == "gcn":
        return GCN(dataset.num_node_features, cfg.hidden_channels, dataset.num_classes)
    elif name == "gcn_regression":
        return GCNNodeRegression(dataset.num_node_features, cfg.hidden_channels, 1)
    elif name == "gcn_scratch":
        return GCNFromScratch(dataset.num_node_features, cfg.hidden_channels, dataset.num_classes)
    else:
        raise ValueError(f"Unknown model type: {name}")


def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    if isinstance(loss_fn, nn.NLLLoss):
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    else:
        loss = loss_fn(out[data.train_mask].squeeze(), data.y[data.train_mask].float())

    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, loss_fn, task_type="classification"):
    model.eval()
    with torch.no_grad():
        out = model(data)

        if task_type == "classification":
            pred = out.argmax(dim=1)
            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
            metrics = compute_classification_metrics(pred[data.test_mask], data.y[data.test_mask])
            return accs, metrics

        elif task_type == "regression":
            losses = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                loss = loss_fn(out[mask].squeeze(), data.y[mask].float())
                losses.append(loss.item())
            metrics = compute_regression_metrics(out[data.test_mask].squeeze(), data.y[data.test_mask])
            return losses, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate GNN Models")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gcn_regression", "gcn_scratch"])
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    cfg = Config()
    dataset, data = load_dataset(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model, cfg, dataset).to(device)
    data = data.to(device)

    if args.model == "gcn_regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"Training {args.model.upper()} on {args.dataset} dataset...")
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer, loss_fn)
        if epoch % 10 == 0:
            if args.model == "gcn_regression":
                losses, _ = evaluate(model, data, loss_fn, "regression")
                print(f"Epoch {epoch:03d} | Train: {losses[0]:.4f} | Val: {losses[1]:.4f} | Test: {losses[2]:.4f}")
            else:
                accs, _ = evaluate(model, data, loss_fn, "classification")
                print(f"Epoch {epoch:03d} | Train: {accs[0]:.4f} | Val: {accs[1]:.4f} | Test: {accs[2]:.4f}")
