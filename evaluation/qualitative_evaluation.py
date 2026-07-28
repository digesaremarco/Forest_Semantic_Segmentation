"""
This module evaluates the trained model on real-world images
stored inside the test_images directory

The evaluation consists of:
    - loading the trained checkpoint
    - running inference
    - displaying predictions
"""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dataset.transforms import ForestTransforms
from evaluation.plots import Plots
from training.checkpoint import Checkpoint

class QualitativeEvaluation:

    def __init__(self, model, device, test_images_dir,  save_directory=None):
        """
        model: trained model
        test_images_dir: directory containing the test images
        save_directory: directory to save the predictions
        """

        self.model = model
        self.device = device
        self.test_images_dir = Path(test_images_dir)
        self.save_directory = Path(save_directory) if save_directory else None

        self.transforms = ForestTransforms().get_transforms()

        if self.save_directory:
            self.save_directory.mkdir(parents=True, exist_ok=True)

        self.plots = Plots(save_directory=self.save_directory)
        self.checkpoint = Checkpoint()


    def load_model(self):
        """
        Load the trained model from the checkpoint
        """

        self.checkpoint.load_checkpoint(self.model)
        self.model.eval()

    def load_images(self, number_of_images=3):
        """
        Load images from the test_images directory
        """

        images = sorted(self.test_images_dir.glob("*"))
        images = [ image for image in images if image.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if len(images) < number_of_images:
            raise ValueError(f"Not enough images in {self.test_images_dir}. Found {len(images)}, but need at least {number_of_images}.")

        return images[:number_of_images]

    def predict(self, image_path):
        """
        Run inference on a single image
        """

        image = np.array(Image.open(image_path).convert("RGB"))
        image_tensor = self.transforms(image=image)["image"]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            logits = output.logits
            logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
            prediction = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype("uint8")

        return np.array(image), prediction


    def evaluate_multiple(self, number_of_images=3):
        """
        Perform qualitative evaluation
        """

        self.load_model()
        images_path = self.load_images(number_of_images=number_of_images)

        images = []
        predictions = []

        for image_path in images_path:
            image, prediction = self.predict(image_path)
            images.append(image)
            predictions.append(prediction)

        self.plots.show_prediction_grid(images, predictions) #save_path=self.save_directory / "qualitative_evaluation.png" if self.save_directory else None)

    def evaluate_single(self, image_path):
        """
        Perform qualitative evaluation on a single image

        image_path: path to the image to evaluate
        """

        self.load_model()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image, prediction = self.predict(image_path)
        self.plots.show_prediction(image, prediction)
