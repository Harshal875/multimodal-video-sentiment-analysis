import torch
from torch.utils.data import DataLoader, Dataset
from models import MultimodalSentimentModel, MultimodalTrainer


class MockDataset(Dataset):
    """Mock dataset that mimics the structure of MELDDataset"""
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'text_inputs': {
                'input_ids': torch.randint(0, 1000, (128,)),  # Sequence length 128
                'attention_mask': torch.ones(128)
            },
            'video_frames': torch.randn(30, 3, 224, 224),  # 30 frames, 3 channels, 224x224
            'audio_features': torch.randn(1, 64, 300),  # Audio mel spectrogram
            'emotion_label': torch.randint(0, 7, (1,)).item(),  # 7 emotion classes
            'sentiment_label': torch.randint(0, 3, (1,)).item()  # 3 sentiment classes
        }


def collate_fn(batch):
    """Custom collate function that matches the one in meld_dataset.py"""
    batch = list(filter(None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def test_logging():
    print("Creating mock datasets...")
    
    # Create mock datasets
    mock_train_dataset = MockDataset(size=50)
    mock_val_dataset = MockDataset(size=20)
    
    # Create data loaders
    mock_train_loader = DataLoader(
        mock_train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    mock_val_loader = DataLoader(
        mock_val_dataset, 
        batch_size=4, 
        collate_fn=collate_fn
    )
    
    print("Creating model...")
    model = MultimodalSentimentModel()
    
    print("Creating trainer...")
    trainer = MultimodalTrainer(model, mock_train_loader, mock_val_loader)
    
    # Test logging functionality
    print("\nTesting logging...")
    
    # Simulate training metrics
    train_losses = {
        'total': 2.5,
        'emotion': 1.0,
        'sentiment': 1.5
    }
    
    trainer.log_metrics(train_losses, phase="train")
    print("✓ Training metrics logged")
    
    # Simulate validation metrics
    val_losses = {
        'total': 1.5,
        'emotion': 0.5,
        'sentiment': 1.0
    }
    val_metrics = {
        'emotion_precision': 0.65,
        'emotion_accuracy': 0.75,
        'sentiment_precision': 0.85,
        'sentiment_accuracy': 0.95
    }
    
    trainer.log_metrics(val_losses, val_metrics, phase="val")
    print("✓ Validation metrics logged")
    
    # Test a forward pass
    print("\nTesting forward pass...")
    model.eval()
    
    for batch in mock_train_loader:
        if batch is not None:
            device = next(model.parameters()).device
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            
            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)
                print(f"✓ Forward pass successful")
                print(f"  - Emotion output shape: {outputs['emotions'].shape}")
                print(f"  - Sentiment output shape: {outputs['sentiments'].shape}")
            break
    
    print("\n✓ All tests passed!")
    print(f"TensorBoard logs saved to: {trainer.writer.log_dir}")
    
    # Close the tensorboard writer
    trainer.writer.close()


if __name__ == "__main__":
    test_logging()