import torch
from torch.autograd import Function
from .._ext import edge_computation


class EdgeComputationFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        assert input.is_cuda

        bs, dim, height, width = input.size()
        output = input.new(bs, 1, height, width).zero_()

        input = torch.sum(input, dim=1, keepdim=True)
        edge_computation.edge_computation_forward_cuda(input, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # print("----------Debugging----------")
        # print(input.size())
        bs, dim, height, width = grad_output.size()
        grad_input = input.new(bs, dim, height, width).zero_()

        edge_computation.edge_computation_backward_cuda(input, grad_output, grad_input)
        grad_input = grad_input.expand_as(input)

        return grad_input
