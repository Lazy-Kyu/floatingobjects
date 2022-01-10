
import torch
from visualization import plot_batch
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, roc_curve, jaccard_score, confusion_matrix
import numpy as np
import pandas as pd


def calculate_metrics(targets, scores, optimal_threshold):
    predictions = scores > optimal_threshold

    auroc = roc_auc_score(targets, scores)
    p, r, f, s = precision_recall_fscore_support(y_true=targets,
                                                 y_pred=predictions, zero_division=0, average="binary")
    kappa = cohen_kappa_score(targets, predictions)
    cm = confusion_matrix(targets, predictions)
    tn = cm[0, 0]
    tp = cm[1, 1]
    fn = cm[0, 1]
    fp = cm[1, 0]
    jaccard = jaccard_score(targets, predictions)

    summary = dict(
        auroc=auroc,
        precision=p,
        recall=r,
        fscore=f,
        kappa=kappa,
        tn=tn,
        tp=tp,
        fn=fn,
        fp=fp,
        jaccard=jaccard,
        threshold=optimal_threshold
    )

    return summary


def snapshot(filename, model, optimizer, epoch, logs):
    torch.save(dict(model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    epoch=epoch,
                    logs=logs),
               filename)


def resume(filename, model, optimizer=None):
    snapshot_file = torch.load(filename)
    model.load_state_dict(snapshot_file["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(snapshot_file["optimizer_state_dict"])
    return snapshot_file["epoch"], snapshot_file["logs"]


def compute_class_occurences(train_loader):
    sum_no_floating, sum_floating = 0, 0
    for idx, (_, target, _) in enumerate(train_loader):
        sum_no_floating += torch.sum(target == 0)
        sum_floating += torch.sum(target == 1)
    return sum_no_floating, sum_floating


def predict_images(dataset, model, device, N_images=5, seed=0):

    # get random images from dataset
    idxs = np.random.RandomState(seed).randint(0, len(dataset), size=N_images)

    # build batch
    data_tuples = [dataset[ix] for ix in idxs]
    images, masks, id = list(map(list, zip(*data_tuples)))
    images, masks = torch.stack(images), torch.stack(masks)

    # predict
    logits = model(images.to(device)).squeeze(1)
    y_preds = torch.sigmoid(logits).detach().cpu().numpy()

    return plot_batch(images, masks, y_preds, id)


def get_scores(val_loader, model, device, n_batches=5, criterion=None):
    y_preds = []
    targets = []
    loss = []
    with torch.no_grad():
        for i in range(n_batches):
            images, masks, id = next(iter(val_loader))
            logits = model(images.to(device)).squeeze(1)

            if criterion is not None:
                valid_data = images.sum(1) != 0  # all pixels > 0
                loss.append(criterion(logits.squeeze(1), masks.to(device), mask=valid_data.to(device)).cpu().numpy())

            y_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(masks.detach().cpu().numpy())

    if criterion is not None:
        return np.vstack(y_preds), np.vstack(targets), np.vstack(loss).mean()
    else:
        return np.vstack(y_preds), np.vstack(targets)

def print_resultscsv(resultscsv, file=None):

    formatters = {
        "auroc":'{:,.2f}'.format,
        "precision":'{:,.2f}'.format,
        "recall":'{:,.2f}'.format,
        "fscore":'{:,.2f}'.format,
        "kappa":'{:,.2f}'.format,
        "jaccard":'{:,.2f}'.format
    }

    df = pd.read_csv(resultscsv, index_col=0).set_index("seed")
    df = df[formatters.keys()]

    print(f"results from {resultscsv}", file=file)
    print("", file=file)
    print(df.to_string(formatters=formatters), file=file)
    print("", file=file)
    print("averaged over seeds", file=file)
    [print(f"{k:<15}: " + v(df.mean(0)[k]) + " +- " + v(df.std(0)[k]), file=file) for k, v in formatters.items()]
