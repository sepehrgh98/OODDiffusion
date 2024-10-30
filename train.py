from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
import lpips
from tqdm import tqdm



from model.ResNet import resnet50
from dataset.HybridDataset import HybridDataset
from utils import instantiate_from_config

import os


def main(args) -> None:

    # Setup accelerator:
    accelerator = Accelerator(split_batches=True, cpu=True)
    set_seed(231)
    # device = accelerator.device
    device = torch.device('cpu')
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")


    # Create model:
    model = resnet50().to(device)

    if cfg.train.resume:
        model.load_state_dict(torch.load(cfg.train.resume, map_location="cpu"), strict=True)
        if accelerator.is_local_main_process:
            print(f"strictly load weight from checkpoint: {cfg.train.resume}")
    else:
        if accelerator.is_local_main_process:
            print("initialize from scratch")



    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Setup data
    dataset: HybridDataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True)
    
    val_dataset: HybridDataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False
    )

    if accelerator.is_local_main_process:
        print(f"Train dataset contains {len(dataset)}")
        print(f"Val dataset contains {len(val_dataset)}")


    # Prepare models for training:
    model, optimizer, loader, val_loader = accelerator.prepare(model, optimizer, loader, val_loader)

    # Setup TensorBoard writer:
    if accelerator.is_local_main_process:
        writer = SummaryWriter(log_dir=os.path.join(exp_dir, "logs"))


    
    
    # Training loop:
    for epoch in range(cfg.train.num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training phase:
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation phase:
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val


        # Log metrics to TensorBoard:
        if accelerator.is_local_main_process:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)


        # Print epoch summary:
        if accelerator.is_local_main_process:
            print(f"Epoch [{epoch+1}/{cfg.train.num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            

    # Close the writer:
    if accelerator.is_local_main_process:
        writer.close()




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)