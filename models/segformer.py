"""
SegFormer model for semantic segmentation

This module provides a wrapper around the Hugging Face SegFormer
implementation for the Forest Semantic Segmentation dataset
"""

import torch
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
)

from models.model_config_loader import (
    BACKBONE,
    NUM_CLASSES,
    PRETRAINED,
    DEVICE,
    IGNORE_INDEX,
    SEGFORMER_MODELS,
)

class SegFormer:

    def __init__(self):

        self.backbone = BACKBONE
        self.num_classes = NUM_CLASSES
        self.pretrained = PRETRAINED
        self.ignore_index = IGNORE_INDEX
        self.segformer_model = SEGFORMER_MODELS

        if DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = DEVICE

        self.model = self.build_model()


    def build_model(self):
        """
        Build the SegFormer model using the specified backbone and number of classes

        return: SegFormer model or error
        """

        if self.backbone not in self.segformer_model:
            raise ValueError(f"Backbone {self.backbone} is not supported. "
                             f"Supported backbones: {list(self.segformer_model.keys())}")

        model_name = self.segformer_model[self.backbone]

        if self.pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(
                model_name,
                num_labels=self.num_classes,
            )
            model = SegformerForSemanticSegmentation(config)

        model.config.ignore_index = self.ignore_index
        model.to(self.device)

        return model

    def get_model(self):
        """
        Return the SegFormer model
        """

        return self.model

    def show_info(self):
        """
        Show information about the SegFormer model
        """

        print(f"Backbone       : {self.backbone.upper()}")
        print(f"Classes        : {self.num_classes}")
        print(f"Pretrained     : {self.pretrained}")
        print(f"Ignore index   : {self.ignore_index}")
        print(f"Device         : {self.device}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Parameters     : {total_params:,}")
        print(f"Trainable      : {trainable_params:,}")


    def summary(self):
        """
        Print a summary of the SegFormer model architecture
        """

        print(self.model)