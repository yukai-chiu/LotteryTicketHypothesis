# LotteryTicketHypothesis
This repository contains a Pytorch implementation the lottery ticket hypothesis
Our team is interested in validating this hypothesis. 
In our ﬁrst task, we will uses a ResNet-18 architecture that has been trained using MNIST dataset. 
Further, we validated the hypothesis on more difﬁcult problems like depth estimation. Current methods for depth estimation use complex encoder-decoder networks which makes it slow for real-time inference. Inference time for depth estimation could be further reduced by using pruning techniques. Fast Depth [5] uses NetAdapt [6] pruning technique for depth estimation. We incorporated Lottery Ticket Hypothesis with Fast Depth network architecture and compare the performance with the original paper using NYUdepthV2dataset.
