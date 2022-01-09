import argparse
from data import FloatingSeaObjectDataset
from transforms import get_transform
import os
import json
from utils import get_scores, resume
from model import get_model
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, \
    cohen_kappa_score, confusion_matrix, jaccard_score


from torch.utils.tensorboard import SummaryWriter
from utils import predict_images, calculate_metrics


def main(args):
    data_path = args.data_path
    args.snapshot_path = os.path.join(args.results_dir, f"model_{args.seed}.pth.tar")


    args_dict = vars(args)

    #### load args from stored JSON file in results dir
    run_arguments_json_file = os.path.join(args.results_dir, f"run_arguments_{args.seed}.json")
    with open(run_arguments_json_file) as json_file:
        run_arguments = json.load(json_file)

    # merge arguments and overwrite stored run arguments if necessary
    for key, value in args_dict.items():
        if value is not None:
            run_arguments[key] = value

    # build Namespace again
    args = argparse.Namespace(**run_arguments)

    summarywriter = SummaryWriter(log_dir=args.tensorboard_logdir) if args.tensorboard_logdir is not None else None

    image_size = args.image_size
    dataset = FloatingSeaObjectDataset(data_path, fold="test", transform=get_transform("test", add_fdi_ndvi=args.add_fdi_ndvi),
                                       output_size=image_size, seed=args.seed, cache_to_npy=args.cache_to_numpy)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    val_dataset = FloatingSeaObjectDataset(data_path, fold="val", transform=get_transform("test", add_fdi_ndvi=args.add_fdi_ndvi),
                                       output_size=image_size, seed=args.seed, cache_to_npy=args.cache_to_numpy)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    inchannels = 12 if not args.add_fdi_ndvi else 14
    model = get_model(args.model, inchannels=inchannels, pretrained=not args.no_pretrained).to(args.device)

    epoch, logs = resume(args.snapshot_path, model)
    print(f"loaded model from snapshot {args.snapshot_path} at epoch {epoch}")

    print("predicting validation dataset")
    valscores, valtargets = get_scores(val_loader, model=model, device=args.device, n_batches=len(val_loader))

    valtargets = valtargets.reshape(-1).astype(int)
    valscores = valscores.reshape(-1)

    fpr, tpr, thresholds = roc_curve(valtargets, valscores)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    optimal_threshold = thresholds[ix]

    print(f'From validation dataset: best threshold={optimal_threshold:.2f}, G-Mean={gmeans[ix]:.4f} with fpr={fpr[ix]:.4f} and tpr={tpr[ix]:.4f}')

    valsummary = calculate_metrics(valtargets, valscores, optimal_threshold)

    print(f"Validation Results")
    print(", ".join([f"{k}={v:.2f}" for k,v in valsummary.items()]))

    valresults_jsonfile = os.path.join(args.results_dir, f"validation_metrics_{args.seed}.json")
    print(f"writing {valresults_jsonfile}")
    with open(valresults_jsonfile, 'w') as fp:
        json.dump({k:float(v) for k,v in valsummary.items()}, fp)

    # free memory as these arrays be quite large...
    del valscores, valtargets

    print("predicting test dataset")
    scores, targets = get_scores(test_loader, model=model, device=args.device, n_batches=len(test_loader))

    targets = targets.reshape(-1).astype(int)
    scores = scores.reshape(-1)

    summarywriter.add_pr_curve("test", targets, scores, global_step=epoch)

    testsummary = calculate_metrics(targets, scores, optimal_threshold)

    fig = predict_images(dataset, model, args.device, N_images=15)
    summarywriter.add_figure("test-predictions", fig, global_step=epoch)

    print(f"Test Results")
    print(", ".join([f"{k}={v:.2f}" for k,v in testsummary.items()]))

    testresults_jsonfile = os.path.join(args.results_dir, f"test_metrics_{args.seed}.json")
    print(f"writing {testresults_jsonfile}")
    with open(testresults_jsonfile, 'w') as fp:
        json.dump({k:float(v) for k,v in testsummary.items()}, fp)

    return testsummary, valsummary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="/data")
    parser.add_argument('--results-dir', type=str, default="/tmp/floatingobjects")

    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0, help="random seed for train/test region split")
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--cache-to-numpy', action="store_true",
                        help="performance optimization: caches images to npz files in a npy folder within data-path.")
    parser.add_argument('--device', type=str, choices=["cpu", "cuda", None], default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())
