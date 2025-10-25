def accuracy(pred, target, mask):
    pred_labels = pred.argmax(dim=1)
    correct = (pred_labels[mask] == target[mask]).sum().item()
    acc = correct / mask.sum().item()
    return acc
