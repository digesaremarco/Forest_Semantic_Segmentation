"""
This module measures the computational performance of a trained model

Computed metrics:
    - number of parameters
    - trainable parameters
    - inference time
    - latency
    - FPS
    - throughput
    - GPU memory usage
"""

import time
import torch
import torch.nn.functional as F
from debugpy.launcher import output

from training.checkpoint import Checkpoint

class Benchmark:

    def __init__(self, model, dataloader, device):

        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.checkpoint = Checkpoint()


    def load_model(self):
        """
        Load the trained model from the checkpoint
        """

        self.checkpoint.load_checkpoint(self.model)
        self.model.eval()

    def number_of_parameters(self):
        """
        Get the number of parameters in the model
        """

        return sum(p.numel() for p in self.model.parameters())

    def trainable_parameters(self):
        """
        Get the trainable parameters in the model
        """

        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def benchmark(self, warmup=20, repetitions=100):
        """
        Benchmark the model for inference

        warmup: number of warmup steps
        repetitions: number of measurement steps
        return: dictionary with benchmark results
        """

        self.load_model()

        images, _ = next(iter(self.dataloader))
        images = images.to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(repetitions):
                output = self.model(images)
                _ = F.interpolate(output.logits, images.shape[-2:], mode='bilinear', align_corners=False)

        # Benchmark
        if self.device == "cuda":
            torch.cuda.synchronize() # Wait for all kernels in all streams on a CUDA device to complete

        start_time = time.time()

        with torch.no_grad():
            for _ in range(repetitions):
                output = self.model(images)
                _ = F.interpolate(output.logits, images.shape[-2:], mode='bilinear', align_corners=False)

        if self.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        total_time = end_time - start_time
        total_images = repetitions * images.shape[0]


        latency = total_time / repetitions
        fps = 1.0 / latency
        throughput = total_images / total_time # Total number of images processed per second

        if self.device == "cuda":
            allocated_memory = torch.cuda.memory_allocated() / 1024*2 # Convert to MB
            peak_memory =   torch.cuda.max_memory_allocated() / 1024*2
        else:
            allocated_memory = 0.0
            peak_memory = 0.0

        results = {
            "parameters": self.number_of_parameters(),
            "trainable_parameters": self.trainable_parameters(),
            "batch_size": images.shape[0],
            "latency_ms": latency * 1000,
            "fps": fps,
            "throughput": throughput,
            "gpu_memory_mb": allocated_memory,
            "gpu_peak_memory_mb": peak_memory,
        }

        return results

    def show_results(self, results):
        """
        Print benchmark results
        """

        print("\nBenchmark Results:\n")

        print(f"Parameters           : {results['parameters']:,}")
        print(f"Trainable            : {results['trainable_parameters']:,}")
        print(f"Batch size           : {results['batch_size']}")
        print(f"Latency              : {results['latency_ms']:.2f} ms")
        print(f"FPS                  : {results['fps']:.2f}")
        print(f"Throughput           : {results['throughput']:.2f} img/s")
        print(f"GPU Memory           : {results['gpu_memory_mb']:.2f} MB")
        print(f"Peak GPU Memory      : {results['gpu_peak_memory_mb']:.2f} MB")