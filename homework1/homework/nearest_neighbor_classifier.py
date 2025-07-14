import torch

# This file uses GitHub Copilot for code assistance.

class NearestNeighborClassifier:
    """
    A class to perform nearest neighbor classification.
    """

    def __init__(self, x: list[list[float]], y: list[float]):
        """
        Store the data and labels to be used for nearest neighbor classification.
        You do not have to modify this function, but you will need to implement the functions it calls.

        Args:
            x: list of lists of floats, data
            y: list of floats, labels
        """
        self.data, self.label = self.make_data(x, y)
        self.data_mean, self.data_std = self.compute_data_statistics(self.data)
        self.data_normalized = self.input_normalization(self.data)

    @classmethod
    def make_data(cls, x: list[list[float]], y: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Warmup: Convert the data into PyTorch tensors.
        """
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    @classmethod
    def compute_data_statistics(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and standard deviation of the data.
        Both should have a shape [1, D]
        """
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return mean, std

    def input_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input x using the mean and std computed from the data in __init__

        Args:
            x: 1D or 2D tensor shape = [D] or [N, D]

        Returns:
            normalized 2D tensor shape = x.shape
        """
        return (x - self.data_mean) / self.data_std

    def get_nearest_neighbor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find the input x's nearest neighbor and the corresponding label.
        """
        x = self.input_normalization(x)
        dists = torch.norm(self.data_normalized - x, dim=1)
        idx = torch.argmin(dists)
        return self.data[idx], self.label[idx]

    def get_k_nearest_neighbor(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find the k-nearest neighbors of input x from the data.
        """
        x = self.input_normalization(x)
        dists = torch.norm(self.data_normalized - x, dim=1)
        idx = torch.topk(dists, k, largest=False).indices
        return self.data[idx], self.label[idx]

    def knn_regression(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Use the k-nearest neighbors of the input x to predict its regression label.
        """
        _, labels = self.get_k_nearest_neighbor(x, k)
        return labels.mean()
