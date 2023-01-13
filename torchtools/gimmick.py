"""
contains functions that helps speed up
    model training/testing, etc.
"""
import os, sys, random, timeit
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from enum import Enum
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


NO_TB_WRITER = False
try: 
    import torch.utils.tensorboard as tb
except ModuleNotFoundError:
    print("[WARN] torch.utils.tensorboard module not found. Disabling create_writer()")
    NO_TB_WRITER = True

sys.path.append(os.path.dirname(__file__))
from .utils import *


class Optics():
    """
    This class contains helper functions for dealing
        with image-related model processes.
    """
    @staticmethod
    def train_step(model: torch.nn.Module, dataloader: DataLoader, 
                    loss_fn: LossFunction, optimizer: Optimizer, 
                    device=DEVICE) -> Tuple[float, float]:
        """
        One training step. Set the model to `training mode`.

        ### args
            - `model`: torch.nn.Module
            - `dataLoader`: torch.utils.data.DataLoader
            - `loss_fn`: loss function
            - `optimizer`: optimizer for the training
            - `device`: default will detect if gpu is available; if yes, use it, else cpu
        """
        train_loss, train_acc = 0, 0

        
        model.train() # enable training mode
        for _, (X, y) in enumerate(dataloader): # going thru batches
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            z = model(X)  # logits
            # 2. Calculate the loss
            loss = loss_fn(z, y)
            train_loss += loss.item()
            # 3. Optimizer zero grad
            optimizer.zero_grad()
            # 4. backpropagation
            loss.backward()
            # 5. gradient descent
            optimizer.step()

            # Calculate accuracy metric
            prob = torch.softmax(z, dim=1) # probabilities
            y_hat = torch.argmax(prob, dim=1)
            train_acc += (y_hat == y).sum().item() / len(z)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc


    @staticmethod
    def test_step(model: torch.nn.Module, dataloader: DataLoader,
                loss_fn: LossFunction, device=DEVICE):
        """
        One test step. Used in the same epoch after `train_step()`

        ### args
            - `model`: torch.nn.Module
            - `dataLoader`: torch.utils.data.DataLoader
            - `loss_fn`: loss function
            - `device`: default will detect if gpu is available; if yes, use it, else cpu
        """
        test_loss, test_acc = 0, 0

        model.eval()
        with torch.inference_mode():
            for _, (X, y) in enumerate(dataloader): # go thru batches
                # Send data to the target device
                X, y = X.to(device), y.to(device)
                # 1. Forward pass
                z = model(X)
                # 2. Calculate the loss
                loss = loss_fn(z, y)
                test_loss += loss.item()

                # Calculate the accuracy
                y_hat = z.argmax(dim=1) # short-hand argmax  logits -> predicition labels
                test_acc += (y_hat == y).sum().item() / len(y_hat)

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc


    @staticmethod
    def train(model: torch.nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, 
            loss_fn: LossFunction, optimizer: Optimizer, 
            epochs: int = 5, writer=None, device=DEVICE):
        """
        Start training loop. Iterate till reach max `epoch`

        ### args
            - `model`: torch.nn.Module
            - `train_dataloader` & `test_dataloader`: torch.utils.data.DataLoader
            - `loss_fn`: loss function
            - `optimizer`: optimizer for the training
            - `device`: default will detect if gpu is available; if yes, use it, else cpu
        
        ### return
            - a dictionary containing losses, accuracies, elapses\n 
                of all epochs for both train step and test step
            
            `If there is a writer, it will be closed after finishing epochs.`
        """
        di = {
                "train_loss": [], "train_acc": [], 
                "test_loss": [], "test_acc": [], 
                "elapsed": []
            }

        for epoch in range(epochs):
            t1 = timeit.default_timer()
            train_loss, train_acc = Optics.train_step(
                                                        model=model,
                                                        dataloader=train_dataloader,
                                                        loss_fn=loss_fn,
                                                        optimizer=optimizer,
                                                        device=device
                                                    )
            test_loss, test_acc = Optics.test_step(
                                                        model=model, 
                                                        dataloader=test_dataloader, 
                                                        loss_fn=loss_fn, 
                                                        device=device
                                                    )
            t2 = timeit.default_timer()
            elapsed = second_to_standard(t2 - t1)

            print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | Elapsed: {elapsed}")

            di["train_loss"].append(train_loss)
            di["train_acc"].append(train_acc)
            di["test_loss"].append(test_loss)
            di["test_acc"].append(test_acc)
            di["elapsed"].append(elapsed)


            ### New: Experiment tracking ###
            # See SummaryWriter documentation
            if writer:
                writer.add_scalars(main_tag="Loss",
                                tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                                global_step=epoch)
                
                writer.add_scalars(main_tag="Accuracy",
                                tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc},
                                global_step=epoch)

                writer.add_scalars(main_tag="Time",
                                tag_scalar_dict={"elapsed": elapsed},
                                global_step=epoch)
                
                writer.add_graph(model=model,
                                input_to_model=torch.randn(32, 3, 224, 224).to(device))

        # Close the writer
        if writer: writer.close()
        return di


    @staticmethod
    def create_dataloaders(transform: TransformsCompose, batch_size: int,
                        root_folder: str=None, train_dir: str=None, test_dir: str=None,
                        num_workers: int = os.cpu_count()) -> tuple[DataLoader, DataLoader, List[str], Dict[str, int]]:
        """
        Create training and testing DataLoaders.

        ### Folder structure
        root_dir/
            |__train/\n
            |__test/

        ### args
            - `transform`: torchvision transforms to perform on training and testing data.
            - `batch_size`: Number of samples per batch in each of the DataLoaders.
            - `root_folder`: root folder which contains a train/ and a test/ \n
                                if they are not under same root, keep this as None
            - `train_dir`: Path to training directory individually
            - `test_dir`: Path to testing directory individually
            - `num_workers`: An integer for number of workers per DataLoader.

        ### return
            A tuple of (train_dataloader, test_dataloader, class_names).
            Where class_names is a list of the target classes. 
        """
        if not root_folder and train_dir and test_dir:
            train_dir, test_dir = Path(train_dir), Path(test_dir)
        elif root_folder:
            train_dir, test_dir = os.path.join(Path(root_folder), "train"), os.path.join(Path(root_folder), "test")
        else:
            raise ValueError("Please at least provide a root folder or train/test folders individually.")

        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)

        # Get class names
        class_names = train_data.classes
        class_dict  = train_data.class_to_idx

        # Turn images into DataLoaders
        train_dataloader = DataLoader(
                                        train_data,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        pin_memory=True,  # enables faster data transfer to CUDA-enabled GPUs
                                    )

        test_dataloader = DataLoader(
                                        test_data,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True,
                                    )

        return train_dataloader, test_dataloader, class_names, class_dict



    @staticmethod
    def save_model(model: torch.nn.Module, save_path: str, model_name: str):
        """
        Saves a PyTorch model to a target directory.

        ### args
            -`model`: A target PyTorch model to save.
            -`target_dir`: A directory for saving the model to.
            -`model_name`: A filename for the saved model. Should include
                either ".pth" or ".pt" as the file extension.
        """
        # Create target directory
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create model save path
        assert model_name.endswith(".pth") or model_name.endswith( ".pt")
        model_save_path = save_path / model_name

        # Save the model state_dict()
        print(f"[INFO] Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)





try:
    def create_writer(experiment_name: str, model_name: str, additional: Optional[str]="") -> tb.writer.SummaryWriter:
        """
        Creates a torch.utils.tensorboard.writer.SummaryWriter() 
            instance that saves to a specific log_dir.

        log_dir is a combination of 
            runs/{timestamp}/{experiment_name}/{model_name}/{extra}.

        ### args
            - `experiment_name: Name of experiment.
            - `model_name: Name of model.
            - `additional (optional)`: Anything extra to add to the directory. Defaults to None.

        ### return
            torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
        """
        if NO_TB_WRITER: raise ModuleNotFoundError("Install tensorboard module first before using this function.")

        ROOT_DIR = "runs" # SummaryWriter default is "run"

        TS = datetime.now().strftime("%Y-%m-%d")
        print(TS)

        log_dir = os.path.join(ROOT_DIR, TS, experiment_name, model_name, additional if additional else "")
            
        print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
        return tb.writer.SummaryWriter(log_dir=log_dir)
except NameError:
    print("[WARN] tensorboard module not found. cannot use create_writer()")