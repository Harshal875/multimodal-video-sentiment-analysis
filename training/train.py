import os 
import argparse
import torch
import torchaudio
import sys       
import tqdm
import json

from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from install_ffmpeg import install_ffmpeg

# AWS SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get(
    'SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get(
    'SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get(
    'SM_CHANNEL_TEST', "/opt/ml/input/data/test")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


#OKAY SO THESE ABOVE DIRECTORIES ARE BASICALLY THE PATHS WHERE THE DATA, TRAINED MODEL,OUTPUT LOGS ARE STORED IN THE AWS SAGEMAKER ENVIRONMENT.
#These paths like /opt/ml/input/data/..., /opt/ml/model/, and /opt/ml/output/ are:
# ✅ Special cloud storage locations on the EC2 instance (SageMaker training container)
# 🚚 Then What Happens?
# After training is done:

# 🗂 Everything in /opt/ml/model/ gets automatically uploaded to S3.

# 📄 Logs from /opt/ml/output/ can also be sent to CloudWatch or S3 (for monitoring/debugging).

# [SageMaker EC2 Instance]
# │
# ├── /opt/ml/input/data/         ← Training/validation data (copied from S3 before training)
# │
# ├── /opt/ml/model/              ← You save model here → SageMaker uploads it to S3
# │
# └── /opt/ml/output/             ← Logs go here → can be streamed or uploaded


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    # Data directories
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)            
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)                       #all of this tells sagemaker where 
                                                                                            #to save the model after training
                                                 #if we want to save the model in a different directory we can change/override the default value of this arg

    return parser.parse_args()

def main():

    if not install_ffmpeg():
        print("Error: FFmpeg installation failed. Cannot continue training.")
        sys.exit(1)

    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    # Prepare data loaders
    #intuition of loader : suppose we have a training dataset of 1000 samples and we want to train our model in batches of 32 samples.
    # this can be done with the help of a train loader which will load the data in batches of 32 samples.

# Imagine that your training data folder (args.train_dir) looks like this in the cloud:

# /opt/ml/input/data/training/
# │
# ├── train_sent_emo.csv
# ├── train_splits/


    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(
            args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size
    )

    print(f"""Training DSV path: {os.path.join(
        args.train_dir, 'train_sent_emo.csv')}""")
    print(f"""Training video directory: {
          os.path.join(args.train_dir, 'train_splits')}""")
    

    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)  

    best_val_loss = float('inf')                                      #if current validation loss is less than best validation 
                                                                      #loss then we will save the model

    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": []
    }

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # Track metrics
        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

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


        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(
                args.model_dir, "model.pth"))                         # Save the best model to the specified directory
            

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






     