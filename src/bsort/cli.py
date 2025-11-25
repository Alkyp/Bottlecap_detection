import argparse
from .train import run_training
from .infer import run_inference
from tools.relabel_and_augment import main as preprocess_pipeline

def main():
    parser = argparse.ArgumentParser(description="Bottlecap Sorting Pipeline (bsort)")
    parser.add_argument("command", type=str, help="preprocess | train | infer")

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_pipeline()

    elif args.command == "train":
        run_training()

    elif args.command == "infer":
        run_inference()

    else:
        print("Unknown command. Use:")
        print("  bsort preprocess")
        print("  bsort train")
        print("  bsort infer")

if __name__ == "__main__":
    main()
