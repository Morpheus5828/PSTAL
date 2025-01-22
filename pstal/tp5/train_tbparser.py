"""This script contains code to learn classifier of tp5
..moduleauthor::Marius THORRE
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import os, sys

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


def fit(
        model,
        input_train,
        feats_train,
        target_train,
        input_dev,
        feats_dev,
        target_dev,
        nb_epochs,
        batch_size,
        device,
):
    criterion = nn.CrossEntropyLoss()
    optim = Adam(lr=1e-4, params=model.parameters())

    history = {
        "train_loss": [],
        "validation_loss": []
    }

    train_dataset = TensorDataset(input_train, feats_train, target_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = TensorDataset(input_dev, feats_dev, target_dev)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_feats, batch_Y in train_dataloader:
            batch_X = batch_X.to(device).long()
            batch_feats = batch_feats.to(device).long()
            batch_Y = batch_Y.to(device).long()

            logits = model(batch_X, batch_feats)
            loss = criterion(logits, batch_Y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_feats, batch_Y in dev_dataloader:
                batch_X = batch_X.to(device).long()
                batch_feats = batch_feats.to(device).long()
                batch_Y = batch_Y.to(device).long()

                logits = model(batch_X, batch_feats)
                loss = criterion(logits, batch_Y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dev_dataloader)
        history["validation_loss"].append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{nb_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    return model, history


