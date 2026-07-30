"""
Checkpoint utilities

This module provides a wrapper for saving and loading
training checkpoints.
"""

from pathlib import Path
import transformers
import re

import torch

from models.model_config_loader import (
    CHECKPOINT_DIR,
    SAVE_BEST_ONLY,
    CHECKPOINT_FILENAME,
)


class Checkpoint:

    def __init__(self):

        self.checkpoint_dir = CHECKPOINT_DIR
        self.save_best_only = SAVE_BEST_ONLY
        self.filename = CHECKPOINT_FILENAME

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True) # Create the checkpoint directory if it doesn't exist

    def save_checkpoint(self, model, optimizer, scheduler, epoch, best_metric):
        """
        Saves the model checkpoint to the specified directory

        model: (torch.nn.Module) The model to be saved
        optimizer: (torch.optim.Optimizer) The optimizer used for training
        scheduler: (torch.optim.lr_scheduler._LRScheduler) The learning rate scheduler used for training
        epoch: (int) The current epoch number
        best_metric: (float) The best metric value achieved during training
        """

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None, # Handle the case where scheduler is None
            "epoch": epoch,
            "best_metric": best_metric,
        }

        checkpoint_path = self.checkpoint_dir / self.filename
        torch.save(checkpoint, checkpoint_path)


    def convert_segformer_state_dict(self, state_dict):
        """
        Converts the state_dict of a SegFormer model to match the expected format for loading
        """

        converted = {}

        for k, v in state_dict.items():

            # stages -> encoder
            m = re.match(r"segformer\.stages\.(\d+)\.patch_embeddings\.(.*)", k)

            if m:
                stage = m.group(1)
                k = f"segformer.encoder.patch_embeddings.{stage}.{m.group(2)}"

            m = re.match(r"segformer\.stages\.(\d+)\.blocks\.(\d+)\.(.*)", k)

            if m:
                stage = m.group(1)
                block = m.group(2)
                k = f"segformer.encoder.block.{stage}.{block}.{m.group(3)}"

            m = re.match(r"segformer\.stages\.(\d+)\.layer_norm\.(.*)", k)

            if m:
                stage = m.group(1)
                k = f"segformer.encoder.layer_norm.{stage}.{m.group(2)}"

            # rename internals
            k = k.replace(".layernorm_before.", ".layer_norm_1.")
            k = k.replace(".layernorm_after.", ".layer_norm_2.")
            k = k.replace(".attention.q_proj.", ".attention.self.query.")
            k = k.replace(".attention.k_proj.", ".attention.self.key.")
            k = k.replace(".attention.v_proj.", ".attention.self.value.")
            k = k.replace(".attention.o_proj.", ".attention.output.dense.")
            k = k.replace(".attention.sequence_reduction.sequence_reduction.", ".attention.self.sr.")
            k = k.replace(".attention.sequence_reduction.layer_norm.", ".attention.self.layer_norm.")
            k = k.replace(".mlp.fc1.", ".mlp.dense1.")
            k = k.replace(".mlp.fc2.", ".mlp.dense2.")
            k = k.replace("decode_head.linear_projections.", "decode_head.linear_c.")

            converted[k] = v

        return converted

    def load_checkpoint(self, model, optimizer=None, scheduler=None):
        """
        Loads the model checkpoint from the specified directory

        return: (epoch, best_metric) where epoch is the last epoch number and best_metric is the best metric value achieved during training
        """

        checkpoint_path = self.checkpoint_dir / self.filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file '{self.filename}' not found in '{self.checkpoint_dir}'.")

        checkpoint = torch.load(checkpoint_path)

        if transformers.__version__ <= "5.0.0":
            state_dict = checkpoint["model_state_dict"]
            state_dict = self.convert_segformer_state_dict(state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if missing:
                print(f"Missing keys: {len(missing)}")
                for k in missing:
                    print(k)
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
                for k in unexpected:
                    print(k)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint["epoch"]
        best_metric = checkpoint["best_metric"]

        return epoch, best_metric

    def delete_checkpoint(self, filename="checkpoint.pth"):
        """
        Deletes the model checkpoint from the specified directory
        """

        checkpoint_path = self.checkpoint_dir / filename

        if checkpoint_path.exists():
            checkpoint_path.unlink() # Delete the checkpoint file

    def checkpoint_exists(self, filename="checkpoint.pth"):
        """
        Checks if the model checkpoint exists in the specified directory

        return: (bool) True if the checkpoint file exists, False otherwise
        """

        checkpoint_path = self.checkpoint_dir / filename
        return checkpoint_path.exists()

    def list_checkpoints(self):
        """
        Lists all the model checkpoints in the specified directory

        return: (list) A list of checkpoint filenames
        """

        return [f.name for f in self.checkpoint_dir.glob("*.pth")]

    def show_info(self):
        """
        Print checkpoint information
        """

        print("Checkpoint")
        print(f"Directory       : {self.checkpoint_dir}")
        print(f"Save best only  : {self.save_best_only}")

        checkpoints = self.list_checkpoints()
        print(f"Files           : {len(checkpoints)}")
        for checkpoint in checkpoints:
            print(f"  - {checkpoint.name}")