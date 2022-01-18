import argparse
from train import main as train
from test import main as test
import os
import pandas as pd
from utils import print_resultscsv

def main(args):
    testresults = []
    valresults = []

    """
    results_dir = args.results_dir
    metric = "auroc"

    valresults = pd.read_csv(os.path.join(results_dir, "valresults.csv"))
    testresults = pd.read_csv(os.path.join(results_dir, "testresults.csv"))

    import json
    seed = 0
    regions = []
    with open(os.path.join(results_dir, f"run_arguments_{seed}.json")) as f:
        data = json.load(f)
        regions.append(
            {
                "train": data["train_regions"],
                "valid": data["valid_dataset"]
            }
        )
    """

    for seed in range(args.num_seeds):
        args.seed = seed
        args.snapshot_path = os.path.join(args.results_dir, f"model_{seed}.pth.tar")
        if args.tensorboard is not None:
            os.makedirs(args.tensorboard, exist_ok=True)
            args.tensorboard_logdir = os.path.join(args.tensorboard, f"model_{seed}/")
        else:
            args.tensorboard_logdir = None
        os.makedirs(args.results_dir, exist_ok=True)

        if args.mode == "train":
            train(args)

        # TEST Trained Model
        testresults_run, valresults_run = test(args)
        testresults_run["seed"] = seed
        valresults_run["seed"] = seed
        testresults.append(testresults_run)
        valresults.append(valresults_run)

    # write as csv
    pd.DataFrame(testresults).to_csv(os.path.join(args.results_dir, "testresults.csv"))
    pd.DataFrame(valresults).to_csv(os.path.join(args.results_dir, "valresults.csv"))

    # write report to file
    with open(os.path.join(args.results_dir, "report.txt"), "w") as f:
        print("Final Validation Results", file=f)
        print_resultscsv(os.path.join(args.results_dir, "valresults.csv"), file=f)
        print("", file=f)
        print("", file=f)
        print("Final Test Results", file=f)
        print_resultscsv(os.path.join(args.results_dir, "testresults.csv"), file=f)

    # print to console
    with open(os.path.join(args.results_dir, "report.txt")) as f:
        print(f.read())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=["train", "test"])
    parser.add_argument('--results-dir', type=str, default="/tmp/floatingobjects")
    parser.add_argument('--tensorboard', type=str, default=None)

    # train arguments
    parser.add_argument('--data-path', type=str, default="/data")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--num-seeds', type=int, default=5, help="num of random seeds")
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--augmentation-intensity', type=int, default=1, help="number indicating intensity 0, 1 (noise), 2 (channel shuffle)")
    parser.add_argument('--model', type=str, default="unet")
    parser.add_argument('--add-fdi-ndvi', action="store_true")
    parser.add_argument('--cache-to-numpy', action="store_true",
                        help="performance optimization: caches images to npz files in a npy folder within data-path.")
    parser.add_argument('--ignore_border_from_loss_kernelsize', type=int, default=0,
                        help="kernel sizes >0 ignore pixels close to the positive class.")
    parser.add_argument('--no-pretrained', action="store_true")
    parser.add_argument('--pos-weight', type=float, default=1, help="positional weight for the floating object class, large values counteract")

    """
    Add a negative outlier loss to the worst classified negative pixels
    """
    parser.add_argument('--neg_outlier_loss_border', type=int, default=19, help="kernel sizes >0 ignore pixels close to the positive class.")
    parser.add_argument('--neg_outlier_loss_num_pixel', type=int, default=100,
                        help="Extra penalize the worst classified pixels (largest loss) of each pixel. Controls a fraction of total number of pixels"
                             "Only useful with ignore_border_from_loss_kernelsize > 0.")
    parser.add_argument('--neg_outlier_loss_penalty_factor', type=float, default=3, help="kernel sizes >0 ignore pixels close to the positive class.")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(parse_args())
