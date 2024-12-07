"""Script to train model."""

import logging
import os
import statistics
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import tqdm
from absl import app, flags
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.transforms import InterpolationMode, transforms

from load_data import CassavaLeafDataset
from model.attention_head import AttentionHead

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="training.log", encoding="utf-8", level=logging.DEBUG
)

_MODEL_NAME = flags.DEFINE_string(
    "model_name", None, "Model name used for training."
)

_CSV_PATH = flags.DEFINE_string("csv_path", None, "Data csv path.")

_TRAIN_IMAGE_DIR = flags.DEFINE_string(
    "train_image_dir", None, "Training images directory path."
)

_RESULT_DIR = flags.DEFINE_string(
    "result_dir", None, "Directory path to stores the result."
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 4, "Training data batch size."
)

_NUM_EPOCHS = flags.DEFINE_integer(
    "num_epochs", 10, "Number of epoch for training a model."
)

_NUM_CLASS = flags.DEFINE_integer(
    "num_class", 5, "Number of classification class."
)

_IS_ATTENTION = flags.DEFINE_bool(
    "is_attention", None, "Does attention head to attached or not."
)

# Model and default weight mapping dictionary
weights_mapping_dict = {
    "efficientnet_b7": "EfficientNet_B7_Weights.IMAGENET1K_V1"
}


def save_mode(model: nn.Module, result_dir: str):
    """Save model at specific path."""
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    path = os.path.join(result_dir, "model.pth")
    torch.save(model.state_dict(), path)


def load_model(
    model_name: str, is_attention: bool, num_class: int
) -> torch.nn.Module:
    """Load torch model with weights."""
    weights = weights_mapping_dict.get(model_name, "")
    model_constructor = getattr(models, model_name)
    model = model_constructor(weights=weights)
    if is_attention:
        model = model.features
        model.Adjust = nn.Conv2d(
            in_channels=2560, out_channels=512, kernel_size=1
        )
        model.AttentionHead = AttentionHead(num_class=num_class)
    else:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class),
        )
    return model


def validation(model: nn.Module, device: str, dataloader: DataLoader) -> float:
    """Validate the model on validation dataset."""
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))

            # Highest probability is our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()

    # Compute accuracy on all set
    accuracy = 100 * accuracy / total
    return accuracy


def train(
    model_name: str,
    csv_path: str,
    train_dir: str,
    result_dir: str,
    batch_size: int,
    num_epochs: int,
    num_class: int,
    is_attention: bool,
) -> None:
    """Model training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model = load_model(
        model_name=model_name, num_class=num_class, is_attention=is_attention
    )
    model.to(device=device)
    # Define transform.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([224], interpolation=InterpolationMode.BICUBIC),
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
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            loss = criterion(prediction, labels)
            if loss == 0:
                continue

            # TODO: learn
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
            scheduler.step(np.mean(losses))

            if loss <= statistics.mean(losses):
                save_mode(model=model, result_dir=result_dir)
            del loss

        # Check on validation set
        if (epoch % 5 == 0) and (epoch != 0):
            acc = validation(
                model=model, device=device, dataloader=validation_loader
            )
            logging.info(
                f"For a epoch number {epoch} accuracy on validation set is {acc}%."  # noqa: E501
            )
            logging.info(f"For a epoch number {epoch} loss is {losses[-1]}.")


def main(argv: Sequence[str]) -> None:
    """Program starts here."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Call train function
    train(
        model_name=_MODEL_NAME.value,
        csv_path=_CSV_PATH.value,
        train_dir=_TRAIN_IMAGE_DIR.value,
        result_dir=_RESULT_DIR.value,
        batch_size=_BATCH_SIZE.value,
        num_epochs=_NUM_EPOCHS.value,
        num_class=_NUM_CLASS.value,
        is_attention=_IS_ATTENTION.value,
    )


if __name__ == "__main__":
    app.run(main)
