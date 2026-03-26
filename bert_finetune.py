import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_scheduler, AutoConfig
import datasets
from datasets import Dataset, DatasetDict

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Hyperparameters:
    hidden_dropout = 0.5
    attention_dropout = 0.5
    lr = 2e-5
    batch_size = 32
    weight_decay = 0.0003
    warmup_steps = 10
    print("Dropout, hidden and attention:", hidden_dropout, attention_dropout)
    print("Learning rate:", lr)
    print("Batch size:", batch_size)
    print("Warmup steps:", warmup_steps)
    print("Weight decay:", weight_decay, "\n")
    suffix = "bs"+str(batch_size)+"_lr"+str(lr)+"_hiddendropout"+str(hidden_dropout)+"_attentiondropout"+str(attention_dropout)+"_weightdecay"+str(weight_decay)+"_warmup"+str(warmup_steps)

    # Import data:
    df_train = pd.read_csv(os.path.join(os.getcwd(), 'data/train.csv'))
    df_val = pd.read_csv(os.path.join(os.getcwd(), 'data/val.csv'))

    # Import BERTje model:
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
    config = AutoConfig.from_pretrained("GroNLP/bert-base-dutch-cased")
    config.hidden_dropout_prob = hidden_dropout
    config.attention_probs_dropout_prob = attention_dropout
    config.num_labels = len(df_train['Label'].unique())
    model = AutoModelForSequenceClassification.from_pretrained("GroNLP/bert-base-dutch-cased", config=config)

    # Preprocess data:
    # Load data from csv to pandas to HuggingFace Dataset:
    ds_train = Dataset.from_pandas(df_train)
    ds_val = Dataset.from_pandas(df_val)

    data = DatasetDict()
    data['train'] = ds_train
    data['validation'] = ds_val

    def tokenize_speech(speech):
        return tokenizer(speech['Text'], padding='max_length', truncation=True,
                         max_length=512, return_tensors='pt')

    tokenized_data = data.map(tokenize_speech, batched=True)
    tokenized_data = tokenized_data.remove_columns(["Text"])
    tokenized_data = tokenized_data.remove_columns(["Speaker"])
    tokenized_data = tokenized_data.remove_columns(["Party"])
    tokenized_data = tokenized_data.rename_column("Label", "labels")
    tokenized_data.set_format("torch")

    train_dataset = tokenized_data["train"]
    val_dataset = tokenized_data["validation"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_epochs = 25
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer,
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=num_training_steps)

    model.to(device)

    # Train model:
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):

        model.train()
        batch_loss = []
        batch_acc = []
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            batch_loss.append(loss.detach().cpu().item())

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            batch_acc.append((sum(predictions == dict(batch.items())['labels']) / len(predictions)).detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        train_loss.append(np.mean(batch_loss))
        train_acc.append(np.mean(batch_acc))

        model.eval()
        batch_loss = []
        batch_acc = []
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            batch_loss.append(outputs.loss.detach().cpu().item())

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            batch_acc.append((sum(predictions == dict(batch.items())['labels']) / len(predictions)).detach().cpu().numpy())

        val_loss.append(np.mean(batch_loss))
        val_acc.append(np.mean(batch_acc))
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print("\tTrain loss:", train_loss[-1])
        print("\tVal loss:", val_loss[-1])
        print("\tTrain acc:", train_acc[-1])
        print("\tVal acc:", val_acc[-1], end="\n\n")

        model.save_pretrained(os.path.join(os.getcwd(), "weights/params_epoch_"+str(epoch)+"_"+suffix))

    model.save_pretrained(os.path.join(os.getcwd(), "weights/params_"+suffix))

    plot_loss_acc_graphs(train_loss, train_acc, val_loss, val_acc, suffix)


def plot_loss_acc_graphs(train_loss, train_acc, val_loss, val_acc, suffix):
    """
    Plots loss and accuracies of training:

    Parameters:
    train_loss: list of lists per epoch with train losses for each sample
    train_acc: list of lists per epoch with train accuracies for each sample
    val_loss: list of lists per epoch with validation losses for each sample
    val_acc: list of lists per epoch with validation accuracies for each sample
    """
    plt.figure()
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Validation loss')
    plt.title("Loss over training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), "plots/loss_graph"+suffix+".png"))

    plt.figure()
    plt.plot(train_acc, label='Train accuracy')
    plt.plot(val_acc, label='Validation accuracy')
    plt.title("Accuracy over training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), "plots/acc_graph"+suffix+".png"))


if __name__ == '__main__':
    main()
