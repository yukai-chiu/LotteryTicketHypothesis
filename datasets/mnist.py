import torchvision.datasets


class MNIST(torchvision.datasets.MNIST):
    """Wrapper for MNIST dataset from PyTorch"""

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=True
    ):
        super(MNIST, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
