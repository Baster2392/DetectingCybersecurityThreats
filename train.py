import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchmetrics import Accuracy
from neural_network import NeuralNetwork


def main():
    batch_size = 32
    epochs = 10
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(batch_size)

    # Creating neural network
    # Create accuracy metric
    metric = Accuracy(task="binary")
    model = NeuralNetwork()

    # Initializing loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train loop
    for epoch in range(epochs):
        print(f"Running epoch number {epoch + 1}...")
        training_loss = 0.0
        model.train()
        for batch_data, batch_labels in test_dataloader:
            optimizer.zero_grad()
            y_predicted = model(batch_data)
            y_predicted = y_predicted.view(-1)
            loss = criterion(y_predicted, batch_labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        print(f"Epoch {epoch + 1} ended with loss {training_loss}")
        print("Validating...")
        # Validate trained model
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader, 0):
                features, labels = data
                y_predicted = model(features)
                y_predicted = y_predicted.view(-1)
                y_predicted = y_predicted.round()
                metric.update(y_predicted, labels)
        acc = metric.compute()
        print(f"Accuracy on validation data: {acc}")
        metric.reset()

    # Test trained model
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            features, labels = data
            y_predicted = model(features)
            y_predicted = y_predicted.view(-1)
            y_predicted = y_predicted.round()
            metric.update(y_predicted, labels)
    test_accuracy = metric.compute()
    print("------------RESULTS------------")
    print(f"Accuracy on test data: {test_accuracy}")
    metric.reset()
    with torch.no_grad():
        for i, data in enumerate(train_dataloader, 0):
            features, labels = data
            y_predicted = model(features)
            y_predicted = y_predicted.view(-1)
            y_predicted = y_predicted.round()
            metric.update(y_predicted, labels)
    train_accuracy = metric.compute()
    print(f"Accuracy on train data: {train_accuracy}")
    metric.reset()
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            features, labels = data
            y_predicted = model(features)
            y_predicted = y_predicted.view(-1)
            y_predicted = y_predicted.round()
            metric.update(y_predicted, labels)
    val_accuracy = float(metric.compute())
    print(f"Accuracy on validation data: {val_accuracy}")
    metric.reset()


def get_dataloader(batch_size):
    # Load preprocessed data
    train_df = pd.read_csv('data/labelled_train.csv')
    test_df = pd.read_csv('data/labelled_test.csv')
    val_df = pd.read_csv('data/labelled_validation.csv')

    # View the first 5 rows of training set
    train_df.head()

    # Separating features and labels
    x_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]
    x_valid = val_df.iloc[:, 0:-1]
    y_valid = val_df.iloc[:, -1]
    x_test = test_df.iloc[:, 0:-1]
    y_test = test_df.iloc[:, -1]
    print(x_train)

    # Scaling features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    # Casting data to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Creating data loaders and samplers
    dataset = TensorDataset(x_train, y_train)
    sample_weights = [0] * len(dataset)

    # Configuring samplers to balance dataset
    train_weights = [1, 450]    # weights of class 0 and 1
    # Adding weights to every data point
    for idx, (data, label) in enumerate(dataset):
        class_weight = train_weights[torch.tensor(label, dtype=torch.int)]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    dataset = TensorDataset(x_valid, y_valid)
    valid_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader


if "__main__" == __name__:
    main()
