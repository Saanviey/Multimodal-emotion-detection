import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, accuracy_score

from meld_dataset import MELDDataset, collate_fn
from models import MultimodalSentimentModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--test-video-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def evaluate(model, data_loader, device):
    model.eval()
    emotion_preds, emotion_labels = [], []
    sentiment_preds, sentiment_labels = [], []

    with torch.inference_mode():
        for batch in data_loader:
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            outputs = model(text_inputs,
                            batch['video_frames'].to(device),
                            batch['audio_features'].to(device))

            emotion_preds.extend(outputs['emotions'].argmax(dim=1).cpu().numpy())
            emotion_labels.extend(batch['emotion_label'].numpy())
            sentiment_preds.extend(outputs['sentiments'].argmax(dim=1).cpu().numpy())
            sentiment_labels.extend(batch['sentiment_label'].numpy())

    return {
        'emotion_accuracy': accuracy_score(emotion_labels, emotion_preds),
        'emotion_precision': precision_score(emotion_labels, emotion_preds, average='weighted', zero_division=0),
        'sentiment_accuracy': accuracy_score(sentiment_labels, sentiment_preds),
        'sentiment_precision': precision_score(sentiment_labels, sentiment_preds, average='weighted', zero_division=0),
    }


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load saved best model weights
    model = MultimodalSentimentModel().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    test_dataset = MELDDataset(args.test_csv, args.test_video_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=1, collate_fn=collate_fn)

    metrics = evaluate(model, test_loader, device)
    print("\nTest results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
