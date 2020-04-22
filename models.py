import torchvision.models


def resnet18(pretrained=False, progress=True, **kwargs):
    """Wrapper for Resnet18 already implemented in PyTorch"""
    net = torchvision.models.resnet18(
        pretrained=pretrained, progress=progress, **kwargs)
    return (net)
