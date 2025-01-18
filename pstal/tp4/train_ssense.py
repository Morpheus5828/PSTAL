"""This script contains code to learn classifier of tp4
..moduleauthor::Marius THORRE
"""

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


def fit(
        model,
        input_train,
        target_train,
        input_dev,
        target_dev,
        vocab_size,
        nb_epochs,
        batch_size,
        device,
):

    criterion = nn.CrossEntropyLoss()
    optim = Adam(lr=1e-3, params=model.parameters())

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

            print("output.shape =", output.shape)
            print("batch_Y.shape =", batch_Y.shape)

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


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    word_ids = [item['word_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    word_ids = pad_sequence(word_ids, batch_first=True, padding_value=-1)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'word_ids': word_ids,
        'labels': labels
    }


def fit_ft(model, train_loader, dev_loader, nb_epochs, device, learning_rate=1e-4):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignorer les labels padding
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "validation_loss": []}

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

    return model, history






