import pdb

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from MonotonicPosterior.model import BinaryClassifier
from MonotonicPosterior.loss import LossBCEMonotonic
from MonotonicPosterior.loss import LossBCEMonotonicAlpha
import torch.optim as optim
from scipy.stats import trim_mean
import numpy as np
import torch
import copy
import pdb


def computePosteriorFromEnsemble(X, y, alpha=None, test_size=0.2, num_ensemble=10, num_layers=5, width=5,
                     learning_rate=0.001, batch_size=0.2, epochs=500):
    Ensemble = []
    for i in range(num_ensemble):
        model,best_state, last_state = computePosterior(X, y, alpha, test_size, num_layers, width,
                     learning_rate, batch_size, epochs)
        model.load_state_dict(best_state)
        Ensemble.append(model)
    [model.eval() for model in Ensemble]
    modelAvg_Func = lambda x: np.mean(np.hstack([model(x).detach().numpy() for model in Ensemble]), axis=1, keepdims=True)
    modelMedian_Func = lambda x: np.median(np.hstack([model(x).detach().numpy() for model in Ensemble]), axis=1,
                                      keepdims=True)
    modelRobustMean_Func = lambda x: np.apply_along_axis(trim_mean, axis=1, arr=np.hstack([model(x).detach().numpy() for model in Ensemble]), proportiontocut=0.2)

    return modelAvg_Func, modelMedian_Func, modelRobustMean_Func, Ensemble

def computePosterior(X, y, alpha=None, test_size=0.2, num_layers=5, width=5,
                     learning_rate=0.001, batch_size=0.2, epochs=500):
    if type(batch_size) == float:
        batch_size = int(batch_size*X.shape[0])
    input_size = 1
    if test_size > 0.0:
        Testing = True
    else:
        Testing = False
    #pdb.set_trace()
    if Testing:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train = X
        X_test = X
        y_train = y
        y_test = y

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))
    test_loader = DataLoader(test_dataset, batch_size=X_test.shape[0], shuffle=True)

    # Initialize model, loss function, and optimizer
    model = BinaryClassifier(input_size=input_size, width=width, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #patience = 10
    # Training loop
    lossTrain = []
    #epochLoss = []
    #lossTest = []
    best_state = None
    lossTestBest = np.inf
    if alpha is None:
        lossFunc = LossBCEMonotonic()
    else:
        lossFunc = LossBCEMonotonicAlpha(alpha=alpha)
    #decreasing = False
    bestEpoch = 1

    for epoch in range(epochs):
        updates = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            # pdb.set_trace()
            predictions = model(batch_X)
            loss = lossFunc(batch_X, predictions, batch_y)
            lossTrain.append(loss.item())
            with torch.no_grad():
                XX_test = torch.tensor(X_test, dtype=torch.float)
                yy_test = torch.tensor(y_test, dtype=torch.float)
                predictions_test = model(XX_test)
                loss_test = lossFunc(XX_test, predictions_test, yy_test)
                # sp.sortedplot(XX_test.numpy(), predictions_test.numpy())
                # sp.sortedplot(batch_X.numpy(), predictions.numpy())
                # sp.ylim(0,1)
                # sp.show()
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lossTestBest >= loss_test:
                lossTestBest = loss_test
                best_state = copy.deepcopy(model.state_dict())
                bestEpoch = epoch + 1
            updates = updates + 1
        # epochLoss.append(sum(lossTrain[-updates:])/updates)
        # if len(epochLoss)>patience:
        #     diff = np.array(epochLoss[-(patience-1):])- np.array(epochLoss[-patience:-1])
        #     if np.all(diff<=0):
        #         break
        #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    print("Best Epoch: " + str(bestEpoch))
    last_state = copy.deepcopy(model.state_dict())
    # Set up EarlyStopping callback
    # early_stopping = EarlyStopping(
    #     monitor="val_loss",  # Metric to monitor
    #     mode="min",          # Minimize the monitored metric
    #     patience=3,          # Stop after 20 epochs without improvement
    #     verbose=True         # Print messages when stopping
    # )

    # # Trainer
    # trainer = pl.Trainer(
    #     max_epochs=500,          # Maximum number of epochs
    #     callbacks=[early_stopping],  # Add EarlyStopping to callbacks
    # )

    # pdb.set_trace()
    # Train the model
    # trainer.fit(model, train_loader, test_loader)
    return model, best_state, last_state

