import os
import argparse
import sys
import subprocess as sp
import json as pyjson
import torch
import torchaudio
from tqdm import tqdm
import json

from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
# from install_ffmpeg import install_ffmpeg

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", ".")
KAGGLE_CHECKPOINT_DATASET = "saanvie/meld-checkpoints"

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--train-video-dir", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--val-video-dir", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--test-video-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()


def push_checkpoint_to_kaggle(model_dir, dataset_slug, message):
    # Writes dataset metadata and pushes model_dir as a new dataset version.
    # check=False so a failed push (e.g. no internet, quota) never crashes training.
    meta = {
        "title": "meld-checkpoints",
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(os.path.join(model_dir, "dataset-metadata.json"), "w") as f:
        pyjson.dump(meta, f)
    sp.run([
        "kaggle", "datasets", "version",
        "-p", model_dir,
        "-m", message,
        "-r", "zip"
    ], check=False)


def main():

    print("Available audio backends:")
    try:
        print(str(torchaudio.list_audio_backends()))
    except AttributeError:
        print("(backend listing not available in this torchaudio version)")

    args = parse_args()
    # for safer local/kaggle testing
    os.makedirs(args.model_dir, exist_ok=True)  # avoid crash when saving model.pth
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=args.train_csv,
        train_video_dir=args.train_video_dir,
        dev_csv=args.val_csv,
        dev_video_dir=args.val_video_dir,
        test_csv=args.test_csv,
        test_video_dir=args.test_video_dir,
        batch_size=args.batch_size
    )

    print(f"Training CSV path: {args.train_csv}")
    print(f"Training video directory: {args.train_video_dir}")

    model = MultimodalSentimentModel().to(device)

    # Resume from last checkpoint if it exists (survives session drops)
    ckpt_path = os.path.join(args.model_dir, "model.pth")
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    trainer = MultimodalTrainer(model, train_loader, val_loader)

    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": []
    }

    for epoch in tqdm(range(start_epoch, args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # Track metrics
        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

        # Log metrics
        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": train_loss["total"]},
                {"Name": "validation:loss", "Value": val_loss["total"]},
                {"Name": "validation:emotion_precision",
                    "Value": val_metrics["emotion_precision"]},
                {"Name": "validation:emotion_accuracy",
                    "Value": val_metrics["emotion_accuracy"]},
                {"Name": "validation:sentiment_precision",
                    "Value": val_metrics["sentiment_precision"]},
                {"Name": "validation:sentiment_accuracy",
                    "Value": val_metrics["sentiment_accuracy"]},
            ]
        }))

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # Save every epoch (not just best) so a resume never loses progress
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, ckpt_path)

        # Push checkpoint to Kaggle Dataset every 3 epochs for crash-proof persistence
        if (epoch + 1) % 3 == 0 or epoch == args.epochs - 1:
            push_checkpoint_to_kaggle(
                args.model_dir,
                KAGGLE_CHECKPOINT_DATASET,
                f"checkpoint at epoch {epoch}"
            )

    # After training is complete, evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_loss"] = test_loss["total"]

    print(json.dumps({
        "metrics": [
            {"Name": "test:loss", "Value": test_loss["total"]},
            {"Name": "test:emotion_accuracy",
                "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test:sentiment_accuracy",
                "Value": test_metrics["sentiment_accuracy"]},
            {"Name": "test:emotion_precision",
                "Value": test_metrics["emotion_precision"]},
            {"Name": "test:sentiment_precision",
                "Value": test_metrics["sentiment_precision"]},
        ]
    }))


if __name__ == "__main__":
    main()