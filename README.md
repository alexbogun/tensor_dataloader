DataLoader for PyTorch optimized for in-GPU-memory tensor datasets.
Can speed up training from 2x to 20x depending on batch sizes.

Repository also contains example for MNIST training (default PyTorch DataLoader: 72s, FastTensorDataLoader: 14s).

Credit goes to Jesse Mu from this thread: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/5
Thanks to Konstantin Schuerholt for correction of case where batch_size == dataset_len.