import torch
import numpy as np
from metrics.depth_metrics import AverageMeter, Result


class Metrics:
    def __init__(self, dataset, train=True):
        self.dataset = dataset
        self.train = train
        self.batch_losses = []

        if dataset == "MNIST":
            self.total_predictions = 0
            self.correct_predictions = 0
            self.batch_accuracies = []

        elif dataset == "nyudepthv2":
            self.batch_metrics = Result()
            self.average_meter = AverageMeter()

        else:
            raise NotImplementedError("No metrics defined for this dataset")

    def update_metrics(self, outputs, labels, loss):
        batch_predications_len = labels.size(0)
        self.batch_losses.append(loss.item() / batch_predications_len)

        if self.dataset == "MNIST":
            _, predicted = torch.max(outputs.data, 1)
            batch_correct_predictions = (predicted == labels).sum().item()
            batch_acc = (batch_correct_predictions / batch_predications_len) * 100.0
            self.batch_accuracies.append(batch_acc)
            self.total_predictions += batch_predications_len
            self.correct_predictions += batch_correct_predictions

        elif self.dataset == "nyudepthv2":
            self.batch_metrics.evaluate(outputs.data, labels.data)
            self.average_meter.update(self.batch_metrics, batch_predications_len)

        else:
            raise NotImplementedError("No metrics defined for this dataset")

    def write_to_tensorboard(self, writer, iteration):
        if self.dataset == "MNIST":
            if self.train:
                writer.add_scalar(
                    "train/accuracy", self.batch_accuracies[-1], iteration
                )
                writer.add_scalar("train/loss", self.batch_losses[-1], iteration)
            else:
                writer.add_scalar(
                    "validation/accuracy", self.batch_accuracies[-1], iteration
                )
                writer.add_scalar("validation/loss", self.batch_losses[-1], iteration)

        elif self.dataset == "nyudepthv2":
            if self.train:
                writer.add_scalar("train/RMSE", self.batch_metrics.rmse, iteration)
                writer.add_scalar("train/MAE", self.batch_metrics.mae, iteration)
                writer.add_scalar("train/DELTA1", self.batch_metrics.delta1, iteration)
                # REL not compatible with tensorboardX since it can be inf/NAN, if using TensorboardX commment out the next line
                writer.add_scalar("train/REL", self.batch_metrics.absrel, iteration)
                writer.add_scalar("train/Lg10", self.batch_metrics.lg10, iteration)
                writer.add_scalar("train/loss", self.batch_losses[-1], iteration)
            else:
                writer.add_scalar("validation/RMSE", self.batch_metrics.rmse, iteration)
                writer.add_scalar("validation/MAE", self.batch_metrics.mae, iteration)
                writer.add_scalar(
                    "validation/DELTA1", self.average_meter[-1].delta1, iteration
                )
                # REL not compatible with tensorboardX since it can be inf/NAN, if using TensorboardX commment out the next line
                writer.add_scalar(
                    "validation/REL", self.batch_metrics.absrel, iteration
                )
                writer.add_scalar("validation/Lg10", self.batch_metrics.lg10, iteration)
                writer.add_scalar("validation/loss", self.batch_losses[-1], iteration)

        else:
            raise NotImplementedError("No metrics defined for this dataset")

    def get_epoch_metrics(self):
        metrics = {}
        if self.dataset == "MNIST":
            metrics["Accuracy"] = (
                self.correct_predictions / self.total_predictions
            ) * 100.0
            metrics["Loss"] = np.mean(self.batch_losses)
        elif self.dataset == "nyudepthv2":
            metrics["RMSE"] = self.average_meter.average().rmse
            metrics["MAE"] = self.average_meter.average().mae
            metrics["DELTA1"] = self.average_meter.average().delta1
            metrics["REL"] = self.average_meter.average().absrel
            metrics["Lg10"] = self.average_meter.average().lg10
            metrics["Loss"] = np.mean(self.batch_losses)

        else:
            raise NotImplementedError("No metrics defined for this dataset")

        return metrics
