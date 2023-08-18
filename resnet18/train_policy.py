import torch
import torch.nn as nn
from ResNet18.policies.MLP_policy import MLP_policy
import torch.optim as optim
from ResNet18.infrastructure.training_utils import train, save_model, validate, save_prediction_plots
from ResNet18.infrastructure.utils import get_a1_dataset, save_dataset, load_dataset, save_plots, get_a1_dataloader, pretrained_ResNet50, extract_batch_feature


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # The learning and training parameters.
    batch_size = 1024
    epochs = 75
    learning_rate = 0.01

    # # Load a1 demo data.
    # a1_dataset = get_a1_dataset()
    # save_dataset(a1_dataset, name='a1_dataset')

    a1_dataset = load_dataset(name='a1_dataset')
    resnet50 = pretrained_ResNet50(device)

    # Extract Feature data
    # feature_dataset = {x: extract_batch_feature(device, resnet50, a1_dataset[x]) for x in ['part_1', 'part_2']}
    # save_dataset(feature_dataset, name='feature_dataset')

    feature_dataset = load_dataset(name='feature_dataset')
    train_loader, valid_loader = get_a1_dataloader(feature_dataset, batch_size=batch_size)

    # Training Process.
    input_size = 4096
    model = MLP_policy(input_size=input_size, size=4, output_size=2) # input size = 4096, size = 4, output_size = 2
    model.to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    print("-"*100)

    # Optimizer and Loss function.
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            reg=True
        )

        valid_epoch_loss, valid_epoch_acc = validate(
            model,
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

    save_prediction_plots(model, valid_loader, device)

    # Save the loss and accuracy plots.
    save_plots(
        train_acc,
        valid_acc,
        train_loss,
        valid_loss,
        name="MLP Error"
    )




    
    



    