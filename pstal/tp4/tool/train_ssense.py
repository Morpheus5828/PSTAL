"""This script contains code to learn classifier of tp4
..moduleauthor::Marius THORRE
"""

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


def fit(
        model,
        input_train,
        target_train,
        input_dev,
        target_dev,
        vocab_size,
        nb_epochs,
        batch_size,
        learning_rate,
        device,
):

    criterion = nn.CrossEntropyLoss()
    optim = Adam(lr=learning_rate, params=model.parameters(), weight_decay=1e-5)

    history = {
        "train_loss": [],
        "validation_loss": []
    }

    train_dataset = TensorDataset(input_train, target_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = TensorDataset(input_dev, target_dev)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    num_samples = input_train.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_Y in train_dataloader:
            batch_X = batch_X.to(device).float()
            batch_Y = batch_Y.to(device).long()
            output = model(batch_X)

            output = output.view(-1, vocab_size)
            batch_Y = batch_Y.view(-1)

            loss = criterion(output, batch_Y)

            optim.zero_grad()
            loss.backward()

            optim.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_dev_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in dev_loader:
                batch_X = batch_X.to(device).float()
                batch_Y = batch_Y.to(device).long()
                outputs = model(batch_X)
                outputs = outputs.view(-1, vocab_size)
                batch_Y = batch_Y.view(-1)
                mask = batch_Y != 0
                outputs = outputs[mask]
                batch_Y = batch_Y[mask]

                loss = criterion(outputs, batch_Y)
                total_dev_loss += loss.item()

        avg_dev_loss = total_dev_loss / len(dev_loader)

        history["validation_loss"].append(avg_dev_loss)

        print(f"Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {avg_dev_loss:.4f}")

    return model, history


def fit_ft(model, train_loader, dev_loader, nb_epochs, device, learning_rate, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    history = {"train_loss": [], "validation_loss": []}

    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(nb_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)

            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            mask = labels != 0
            outputs = outputs[mask]
            batch_Y = labels[mask]

            optimizer.zero_grad()

            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)

                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)

                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dev_loader)
        history["validation_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    return model, history





