"""Script to train model."""

import logging
from collections.abc import Sequence

import torch
import tqdm
import torch.nn as nn
from absl import app, flags
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.transforms import InterpolationMode, transforms

from data.load_data import CassavaLeafDataset

_MODEL_NAME = flags.DEFINE_string(
    "model_name", None, "Model name used for training."
)

_CSV_PATH = flags.DEFINE_string("csv_path", None, "Data csv path.")

_TRAIN_IMAGE_DIR = flags.DEFINE_string(
    "train_image_dir", None, "Training images directory path."
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 32, "Training data batch size."
)

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs", 10, "Number of epoch for training a model."
)

# Model and default weight mapping dictionary
weights_mapping_dict = {
    "efficientnet_b7": "EfficientNet_B7_Weights.IMAGENET1K_V1"
}

def save_mode(model: nn.Module):
    """Save model at specific path."""
    path = "tmp.pth"
    torch.save(model.state_dict(), path)


def load_model(model_name: str) -> torch.nn.Module:
    """Load torch model with weights."""
    weights = weights_mapping_dict.get(model_name, "")
    model_constructor = getattr(models, model_name)
    model = model_constructor(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(model.classifier[1].in_features, 5)
    )
    return model


def train(
    model_name: str,
    csv_path: str,
    train_dir: str,
    batch_size: int,
    num_epochs: int,
) -> None:
    """Model training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model = load_model(model_name=model_name)
    model.to(device=device)
    # Define transform.
    transform = transforms.Compose(
        [
            transforms.Resize([600], interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    # Set loss,optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, patience=4, verbose=True, factor=0.2
    )  # TODO: learn

    # Load dataset
    dataset = CassavaLeafDataset(
        csv_path=csv_path, root_dir=train_dir, transform=transform
    )
    validation_size = int(0.2 * len(dataset))
    train_size = len(dataset) - validation_size
    train_data, validation_data = random_split(
        dataset, [train_size, validation_size]
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=True
    )
    logging.info(f"Num of training exmaples: {train_size}")
    logging.info(f"Num of validation examples: {validation_size}")

    # Traning loop   
    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        losses = []

        for batch_index, (images, labels) in enumerate(train_loader):
            # Forward
            prediction = model(images.to(device=device))
            loss = criterion(prediction, labels)
            losses.append(loss)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Save model take average of loss list and check with current loss for saving.



def main(argv: Sequence[str]) -> None:
    """Program starts here."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Call train function
    train(
        model_name=_MODEL_NAME.value,
        csv_path=_CSV_PATH.value,
        train_dir=_TRAIN_IMAGE_DIR.value,
        batch_size=_BATCH_SIZE.value,
        num_epochs=_NUM_EPOCHS.value,
    )


if __name__ == "__main__":
    app.run(main)
