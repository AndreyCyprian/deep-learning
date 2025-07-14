import torch
# This file uses GitHub Copilot for code assistance.


class PyTorchBasics:
    """
    Implement the following python code with PyTorch.
    Use PyTorch functions to make your solution efficient and differentiable.

    General Rules:
    - No loops, no function calls (except for torch functions), no if statements
    - No numpy
    - PyTorch and tensor operations only
    - No assignments to results x[1] = 5; return x
    - A solution requires less than 10 PyTorch commands

    The grader will convert your solution to torchscript and make sure it does not
    use any unsupported operations (loops etc).
    """

    @staticmethod
    def make_it_pytorch_1(x: torch.Tensor) -> torch.Tensor:
        """
        Return every 3rd element of the input tensor.

        x is a 1D tensor

        --------
        y = []
        for i, v in enumerate(x):
            if i % 3 == 0:
                y.append(v)
        return torch.stack(y, dim=0)
        --------

        Solution length: 13 characters
        """
        indices = torch.arange(0, x.size(0), 3)
        return torch.index_select(x, 0, indices)

    @staticmethod
    def make_it_pytorch_2(x: torch.Tensor) -> torch.Tensor:
        """
        Return the maximum value of each row of the final dimension of the input tensor

        x is a 3D tensor

        --------
        n, m, _ = x.shape
        y = torch.zeros(n, m)
        for i in range(n):
            for j in range(m):
                maxval = float("-inf")
                for v in x[i, j]:
                    if v > maxval:
                        maxval = v
                y[i, j] = maxval
        return y
        --------

        Solution length: 26 characters
        """
        return torch.max(x, dim=-1).values

    @staticmethod
    def make_it_pytorch_3(x: torch.Tensor) -> torch.Tensor:
        """
        Return the unique elements of the input tensor in sorted order

        x can have any dimension

        --------
        y = []
        for i in x.flatten():
            if i not in y:
                y.append(i)
        return torch.as_tensor(sorted(y))
        --------

        Solution length: 22 characters
        """
        return torch.sort(torch.unique(x))[0]

    @staticmethod
    def make_it_pytorch_4(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the number of elements in y that are greater than the mean of x

        x and y can have any shape
        """
        mean = torch.mean(x)
        return torch.sum(y > mean)

    @staticmethod
    def make_it_pytorch_5(x: torch.Tensor) -> torch.Tensor:
        """
        Return the transpose of the input tensor

        x is a 2D tensor
        """
        return x.t()

    @staticmethod
    def make_it_pytorch_6(x: torch.Tensor) -> torch.Tensor:
        """
        Return the diagonal elements (top left to bottom right) of the input tensor

        x is a 2D tensor
        """
        return torch.diag(x)

    @staticmethod
    def make_it_pytorch_7(x: torch.Tensor) -> torch.Tensor:
        """
        Return the diagonal elements (top right to bottom left) of the input tensor

        x is a 2D tensor
        """
        return torch.diag(torch.fliplr(x))

    @staticmethod
    def make_it_pytorch_8(x: torch.Tensor) -> torch.Tensor:
        """
        Return the cumulative sum of the input tensor

        x is a 1D tensor
        """
        return torch.cumsum(x, dim=0)

    @staticmethod
    def make_it_pytorch_9(x: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of all elements in the rectangle upto (i, j)th element

        x is a 2D tensor
        """
        return torch.cumsum(torch.cumsum(x, dim=0), dim=1)

    @staticmethod
    def make_it_pytorch_10(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Return the input tensor with all elements less than c set to 0

        x is a 2D tensor
        c is a scalar tensor (dimension 0)
        """
        return torch.where(x < c, torch.zeros_like(x), x)

    @staticmethod
    def make_it_pytorch_11(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Return the row and column indices of the elements less than c

        x is a 2D tensor
        c is a scalar tensor (dimension 0)

        The output is a 2 x n tensor where n is the number of elements less than c
        """
        return torch.nonzero(x < c, as_tuple=False).T

    @staticmethod
    def make_it_pytorch_12(x: torch.Tensor, m: torch.BoolTensor) -> torch.Tensor:
        """
        Return the elements of x where m is True

        x and m are 2D tensors
        """
        return x[m]

    @staticmethod
    def make_it_pytorch_extra_1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the difference between consecutive elements of the sequence [x, y]

        x and y are 1D tensors
        """
        xy = torch.cat((x, y))
        return xy[1:] - xy[:-1]

    @staticmethod
    def make_it_pytorch_extra_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Find the number of elements in x that are equal (abs(x_i-y_j) < 1e-3) to at least one element in y

        x and y are 1D tensors
        """
        diff = (x.unsqueeze(1) - y.unsqueeze(0)).abs() < 1e-3
        return torch.sum(diff.any(dim=1))
