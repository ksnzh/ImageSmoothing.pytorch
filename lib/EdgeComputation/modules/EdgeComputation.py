from torch.nn.modules.module import Module
from ..functions.EdgeComputation import EdgeComputationFunction


class EdgeComputationModule(Module):
    def __init__(self):
        super(EdgeComputationModule, self).__init__()

    def forward(self, input):
        return EdgeComputationFunction.apply(input)
