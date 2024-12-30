import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
import copy
import pdb




class BinaryClassifier(nn.Module):
    def __init__(self, input_size, width, num_layers):
        super(BinaryClassifier, self).__init__()
        layers = []
        in_features = input_size

        for i in range(num_layers):
            out_features = width if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                # layers.append(nn.BatchNorm1d(out_features))
                #layers.append(nn.LeakyReLU())
                layers.append(nn.GELU())
                #layers.append(nn.ELU())
                #layers.append(nn.SELU())
                # layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        if num_layers > 0:
            layers.append(nn.Tanh())  # Output layer for binary classification

        self.network = nn.Sequential(*layers)
        # self.criterion = LossBCEMonotonic()
        # pdb.set_trace()

    def forward(self, x):
        # pdb.set_trace()
        x = self.network(x)
        x = (x + 1) / 2
        return x

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     #pdb.set_trace()
    #     y_pred = self(x)
    #     loss = self.criterion(x, y_pred, y)
    #     self.log("train_loss", loss)  # Logging for training
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     #pdb.set_trace()
    #     y_pred = self(x)
    #     val_loss = self.criterion(x, y_pred, y)
    #     self.log("val_loss", val_loss)  # Logging for validation
    #     return val_loss

    # def configure_optimizers(self):
    #     return torch.optim.SGD(self.parameters(), lr=0.01)


