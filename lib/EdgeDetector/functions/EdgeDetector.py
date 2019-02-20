import torch
from torch.autograd import Function
from .._ext import edge_detector


class EdgeDetectorFunction(Function):
    def __init__(self, isSmoothing):
        self.isSmoothing = isSmoothing

    def forward(self, input):
        assert input.is_cuda

        input_image = input[:, 0:3, :, :]
        input_edge = input[:, 3:4, :, :]

        bs, dim, height, width = input_edge.size()
        self.output_preserve = input.new(bs, dim, height, width).zero_()
        self.output_eliminate = input.new(bs, dim, height, width).zero_()
        # self.output = input.new(bs, dim * 2, height, width).zero_()

        input_image = torch.sum(input_image, dim=1, keepdim=True)

        edge_detector.edge_detector_cuda(
            input_image,
            input_edge,
            self.output_preserve,
            self.output_eliminate,
            self.isSmoothing
        )

        # self.output[:, 0, :, :] = self.output_preserve
        # self.output[:, 1, :, :] = self.output_preserve
        self.output = torch.cat([self.output_preserve, self.output_eliminate], dim=1)

        return self.output

    def backward(self, grad_output):
        return None
