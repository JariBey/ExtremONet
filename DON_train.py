import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import time
import torch
import torch.nn as nn

def train_DON(model, optimizer, scheduler, train_data, iterations, test_split=0.2, device=None, nprint=10):
    x, u, y = train_data
    x, u, y = torch.tensor(x, dtype=model.dtype).to(model.device), torch.tensor(u, dtype=model.dtype).to(model.device), torch.tensor(y, dtype=model.dtype).to(model.device)
    model.branch.norm = [torch.mean(u).item(), torch.std(u).item()]
    model.trunk.norm = [torch.mean(x).item(), torch.std(x).item()]
    
    # Split data into training and testing sets
    num_samples = x.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(np.floor(test_split * num_samples))
    train_indices, test_indices = indices[split:], indices[:split]
    x_train, u_train, y_train = x[train_indices], u[train_indices], y[train_indices]
    x_test, u_test, y_test = x[test_indices], u[test_indices], y[test_indices]

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_train_loss = float('inf')
    best_model_state = None

    start_time = time.time()
    for epoch in range(iterations):
        model.train()
        optimizer.zero_grad()
        y_pred = model.forward(x_train, u_train)
        train_loss = model.loss(y_train, y_pred)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            y_test_pred = model.forward(x_test, u_test)
            test_loss = model.loss(y_test, y_test_pred)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            best_train_loss = train_loss.item()
            best_model_state = model.state_dict()

        if (epoch * nprint) % iterations == 0:
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (epoch + 1) * (iterations - epoch - 1)
            print('----------------------------------------------------------------------------------------------------------------------------------------')
            print(f'Epoch {epoch + 1}/{iterations}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}, ETA: {eta:.2f}s')

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f'Lowest Test Loss: {best_test_loss:.6f}, Corresponding Train Loss: {best_train_loss:.6f}')

    return train_losses, test_losses