"""This script contains code to train tp3 model
..moduleauthor:: Marius THORRE
"""

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from model import MorphModel, MultiTaskMorphModel


def fit(
        train_data,
        dev_data,
        vocab_size_input,
        vocab_size_output,
        nb_epochs,
        batch_size,
        device,
        embedding_size,
        hidden_layer,
        hidden_dim
):
    model = MorphModel(
        vocab_size_input=vocab_size_input,
        vocab_size_output=vocab_size_output,
        d=hidden_dim,
        embedding_size=embedding_size,
        hidden_layer=hidden_layer
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=2)
    optim = Adam(lr=1e-4, params=model.parameters())

    history = {
        "train_loss": [],
        "validation_loss": []
    }

    train_dataset = TensorDataset(train_data["in_enc_train"], train_data["ends_train"], train_data["feats_number_train"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(dev_data["in_enc_dev"], dev_data["ends_dev"], dev_data["feats_number_dev"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_ends, batch_Y in train_dataloader:
            batch_X = batch_X.to(device).long()
            batch_ends = batch_ends.to(device).long()
            batch_Y = batch_Y.to(device).long()

            outputs = model(batch_X)

            batch_ends_expanded = batch_ends.unsqueeze(2).expand(-1, -1,
                                                                 outputs.size(2))
            gathered = outputs.gather(1, batch_ends_expanded)

            gathered = gathered.view(-1, vocab_size_output)
            batch_Y = batch_Y.view(-1)

            loss = criterion(gathered, batch_Y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_X, batch_ends, batch_Y in test_loader:
                batch_X = batch_X.to(device).long()
                batch_ends = batch_ends.to(device).long()
                batch_Y = batch_Y.to(device).long()

                outputs = model(batch_X)
                batch_ends_expanded = batch_ends.unsqueeze(2).expand(-1, -1, outputs.size(2))
                gathered = outputs.gather(1, batch_ends_expanded)
                gathered = gathered.view(-1, vocab_size_output)
                batch_Y = batch_Y.view(-1)

                loss = criterion(gathered, batch_Y)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        history["validation_loss"].append(avg_test_loss)

        print(f"\t Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
    }, 'model_simple_task.pt')


    return model, history


def fit_multitask(
        train_data,
        dev_data,
        vocab_size_input,
        all_feats_dict,
        nb_epochs,
        batch_size,
        device,
        embedding_size,
        hidden_layer,
        hidden_dim
):
    model = MultiTaskMorphModel(
        vocab_size_input=vocab_size_input,
        all_feats_dict=all_feats_dict,
        d=hidden_dim,
        embedding_size=embedding_size,
        hidden_layer=hidden_layer
    ).to(device)

    criterions = {tm: nn.CrossEntropyLoss(ignore_index=2) for tm in all_feats_dict.keys()}
    optimizer = Adam(model.parameters(), lr=1e-4)

    history = {"train_loss": [], "validation_loss": []}

    train_dataset = TensorDataset(train_data["in_enc"], train_data["ends"], *[train_data["feats_dict"][tm] for tm in all_feats_dict.keys()])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = TensorDataset(dev_data["in_enc"], dev_data["ends"], *[dev_data["feats_dict"][tm] for tm in all_feats_dict.keys()])
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(nb_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            batch_X = batch[0].to(device).long()
            batch_ends = batch[1].to(device).long()
            batch_Ys = {tm: batch[i + 2].to(device).long() for i, tm in enumerate(all_feats_dict.keys())}

            optimizer.zero_grad()
            outputs = model(batch_X)

            loss = torch.tensor(0.0, device=device)
            for tm, output in outputs.items():
                batch_ends_expanded = batch_ends.unsqueeze(2).expand(-1, -1, output.size(2))
                gathered = output.gather(1, batch_ends_expanded)
                gathered = gathered.view(-1, len(all_feats_dict[tm]))
                batch_Y = batch_Ys[tm].view(-1)
                loss += criterions[tm](gathered, batch_Y)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_dev_loss = 0

        with torch.no_grad():
            for batch in dev_loader:
                batch_X = batch[0].to(device).long()
                batch_ends = batch[1].to(device).long()
                batch_Ys = {tm: batch[i + 2].to(device).long() for i, tm in enumerate(all_feats_dict.keys())}

                outputs = model(batch_X)

                loss = torch.tensor(0.0, device=device)
                for tm, output in outputs.items():
                    batch_ends_expanded = batch_ends.unsqueeze(2).expand(-1, -1, output.size(2))
                    gathered = output.gather(1, batch_ends_expanded)
                    gathered = gathered.view(-1, len(all_feats_dict[tm]))
                    batch_Y = batch_Ys[tm].view(-1)
                    loss += criterions[tm](gathered, batch_Y)

                total_dev_loss += loss.item()

        avg_dev_loss = total_dev_loss / len(dev_loader)
        history["validation_loss"].append(avg_dev_loss)

        print(f"\t Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_dev_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
    }, 'model_multi_task.pt')

    return model, history


