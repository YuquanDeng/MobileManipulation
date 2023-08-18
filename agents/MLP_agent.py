import torch.optim as optim
import torch.nn as nn
import sys
sys.path.append("../")
from infrastructure.utils import *
from infrastructure.training_utils import save_prediction_plots, train, validate, save_model, save_model_params
from policies.MLP_policy import MLP_policy


class MLPagent():
    def __init__(self, agent_params):
        self.agent_params = agent_params
        self.model = MLP_policy(input_size=4096, size=4, output_size=2) # input size = 4096, size = 4, output_size = 2
        self.criterion = nn.MSELoss()

    def train(self, train_loader, valid_loader, device):
        # The learning and training parameters.
        epochs = self.agent_params['epochs']
        learning_rate = self.agent_params['learning_rate']
        all_logs = {}

        # Training Process.
        self.model.to(device)

        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")
        print("-"*100)

        # Optimizer and Loss function.
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = self.criterion

        # Lists to keep track of losses and accuracies.
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        # Start the training.
        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")
            train_epoch_loss, train_epoch_acc = train(
                self.model,
                train_loader,
                optimizer,
                criterion,
                device,
                reg=True
            )

            valid_epoch_loss, valid_epoch_acc = validate(
                self.model,
                valid_loader,
                criterion,
                device,
                reg=True
            )

            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_loss.append(valid_epoch_loss)
            valid_acc.append(valid_epoch_acc)
            print(f"Training loss: {train_epoch_loss: .3f}, training acc: {train_epoch_acc: .3f}")
            print(f"Validation loss: {valid_epoch_loss: .3f}, validation acc: {valid_epoch_acc: .3f}")
            print('-'*100)
        
        all_logs['train_loss'] = train_loss
        all_logs['valid_loss'] = valid_loss

        return all_logs.copy()

    def save_model_params(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)
        print('-'*100)
        print(f"Successfully saved model as {filepath}")
    
    def load_model_params(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath))
        print('-'*100)
        print(f"Successfully loaded model as {filepath}")
