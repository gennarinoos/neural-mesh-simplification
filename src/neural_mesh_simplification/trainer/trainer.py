import logging
import os
from typing import Dict, Any

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader

from ..data import MeshSimplificationDataset
from ..losses import CombinedMeshSimplificationLoss
from ..metrics import chamfer_distance, normal_consistency, edge_preservation, hausdorff_distance
from ..models import NeuralMeshSimplification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Initializing trainer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        logger.info("Initializing model...")
        self.model = NeuralMeshSimplification(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            edge_hidden_dim=config["model"]["edge_hidden_dim"],
            num_layers=config["model"]["num_layers"],
            k=config["model"]["k"],
            edge_k=config["model"]["edge_k"],
            target_ratio=config["model"]["target_ratio"],
        ).to(self.device)

        logger.info("Setting up optimizer and loss...")
        self.optimizer = Adam(
            self.model.parameters(), lr=config["training"]["learning_rate"]
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
        self.criterion = CombinedMeshSimplificationLoss(
            lambda_c=config["loss"]["lambda_c"],
            lambda_e=config["loss"]["lambda_e"],
            lambda_o=config["loss"]["lambda_o"],
        )
        self.early_stopping_patience = config["training"]["early_stopping_patience"]
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logger.info("Preparing data loaders...")
        self.train_loader, self.val_loader = self._prepare_data_loaders()
        logger.info("Trainer initialization complete.")

    def _prepare_data_loaders(self):
        logger.info(f"Loading dataset from {self.config['data']['data_dir']}")
        dataset = MeshSimplificationDataset(data_dir=self.config["data"]["data_dir"])
        logger.info(f"Dataset size: {len(dataset)}")

        val_size = int(len(dataset) * self.config["data"]["val_split"])
        train_size = len(dataset) - val_size
        logger.info(f"Splitting dataset: {train_size} train, {val_size} validation")

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        assert len(val_dataset) > 0, \
            f"There is not enough data to define an evaluation set. len(dataset)={len(dataset)}, train_size={train_size}, val_size={val_size}"
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            follow_batch=["x", "pos"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            follow_batch=["x", "pos"],
        )
        logger.info("Data loaders prepared successfully")

        return train_loader, val_loader

    def train(self):
        for epoch in range(self.config["training"]["num_epochs"]):
            self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)
            self.scheduler.step(val_loss)

            # Apply weight decay after each epoch (as per paper)
            weight_decay = self.config["training"]["weight_decay"]
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data *= weight_decay

            self._save_checkpoint(epoch, val_loss)
            if self._early_stopping(val_loss):
                logging.info("Early stopping triggered.")
                break

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        logger.debug(f"Starting epoch {epoch + 1}")
        for batch_idx, batch in enumerate(self.train_loader):
            logger.debug(f"Processing batch {batch_idx + 1}")
            try:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = self.criterion(batch, output)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Batch {batch_idx + 1} - Loss: {loss.item():.4f}")
            except Exception as e:
                logger.error(f"Error in batch {batch_idx + 1}: {str(e)}")
                raise e
        logging.info(
            f"Epoch [{epoch + 1}/{self.config['training']['num_epochs']}], Loss: {running_loss / len(self.train_loader)}"
        )

    def _validate(self, epoch: int) -> float:
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = self.criterion(batch, output)
                val_loss += loss.item()
        val_loss /= len(self.val_loader)
        logging.info(f"Epoch [{epoch + 1}/{self.config['training']['num_epochs']}], Validation Loss: {val_loss}")
        return val_loss

    def _save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            checkpoint_path,
        )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)

    def _early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        return self.early_stopping_counter >= self.early_stopping_patience

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["val_loss"]
        logging.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")

    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        log_message = f"Epoch [{epoch + 1}/{self.config['training']['num_epochs']}], "
        log_message += ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        logging.info(log_message)

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        metrics = {
            "chamfer_distance": 0.0,
            "normal_consistency": 0.0,
            "edge_preservation": 0.0,
            "hausdorff_distance": 0.0
        }
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                output = self.model(batch)

                # TODO: Define methods that can operate on a batch instead of a trimesh object

                metrics["chamfer_distance"] += chamfer_distance(batch, output)
                metrics["normal_consistency"] += normal_consistency(batch, output)
                metrics["edge_preservation"] += edge_preservation(batch, output)
                metrics["hausdorff_distance"] += hausdorff_distance(batch, output)
        for key in metrics:
            metrics[key] /= len(data_loader)
        return metrics

    def handle_error(self, error: Exception):
        logging.error(f"An error occurred: {error}")
        if isinstance(error, RuntimeError) and "out of memory" in str(error):
            logging.error("Out of memory error. Attempting to recover.")
            torch.cuda.empty_cache()
        else:
            raise error

    def save_training_state(self, state_path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "early_stopping_counter": self.early_stopping_counter,
            },
            state_path,
        )

    def load_training_state(self, state_path: str):
        state = torch.load(state_path)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.best_val_loss = state["best_val_loss"]
        self.early_stopping_counter = state["early_stopping_counter"]
        logging.info(f"Loaded training state from {state_path}")
