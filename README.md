Deepfake Detection using XceptionNet

This repository provides a PyTorch implementation for deepfake image classification using the XceptionNet architecture (via timm). The project is built on the Celeb-DF v2 dataset, which contains both real and synthetically generated (fake) celebrity face images.

The goal is to train a binary classifier to distinguish between real and fake images.

🚀 Features

Custom PyTorch Dataset class to handle Celeb-DF images.

Image preprocessing and augmentation with Torchvision Transforms.

Train/validation split with DataLoader for efficient batching.

Model definition using XceptionNet (pretrained on ImageNet via timm).

Binary classification with BCEWithLogitsLoss.

Accuracy and loss tracking for both training and validation sets.

Visualization of training curves (loss & accuracy).

Model saving for later inference or fine-tuning.

📂 Dataset

The project uses the Celeb-DF v2 dataset, with images extracted from the video frames.
Directory structure should look like this:

C:/792/Celeb-DF-v2_images/
│── Celeb-real_images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
└── Celeb-synthesis_images/
    ├── fake1.jpg
    ├── fake2.jpg
    └── ...


Celeb-real_images → contains real celebrity images.

Celeb-synthesis_images → contains fake (deepfake) images.

⚙️ Requirements

Install the following Python packages:

pip install torch torchvision timm opencv-python matplotlib pandas numpy pillow

🛠️ Training Workflow

Dataset Loading

Uses CustomDataset to load images and map them to labels:

0 → Real

1 → Fake

Preprocessing

Resize images to 128x128.

Convert to tensors.

Data Split

80% training

20% validation

Model

XceptionNet (timm.create_model('xception', pretrained=True, num_classes=1)).

Training Loop

Optimizer: Adam with learning rate 1e-4.

Loss: BCEWithLogitsLoss.

Tracks accuracy and loss across epochs.

Visualization

Plot training/validation accuracy and loss vs. epochs.

Model Saving

Saves model state, optimizer state, and training history in CelebDF.pt.

📊 Example Outputs

Training & validation accuracy curves.

Loss vs. epoch curves.

Final saved model (CelebDF.pt).

🐞 Common Issues

num_samples=0 error → occurs when the dataset path is empty or not structured properly.
✅ Ensure C:/792/Celeb-DF-v2_images/ contains Celeb-real_images/ and Celeb-synthesis_images/ folders with .jpg images.

▶️ Usage

Clone the repo and place dataset in the correct directory.

Run the training script (e.g., in Jupyter/Colab).

Plot training results with plot_curves(train_losses, train_accs, val_losses, val_accs).

Use the saved model for inference or fine-tuning.

📌 Next Steps

Add data augmentation for robustness.

Implement test-time evaluation.

Explore lightweight models for real-time inference.

Extend to video-based detection.

📜 License

This project is for research and educational purposes only.
