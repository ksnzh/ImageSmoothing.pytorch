from torch.nn.modules.module import Module
from ..functions.smooth_edge_term_loss import SmoothEdgeTermLossFunction


class SmoothEdgeTermLossModule(Module):
    def __init__(self, sigma_color, sigma_space, window_size, lp, w_smooth, w_edge, w_L2, isDetailEnhancement, isStylization):
        super(SmoothEdgeTermLossModule, self).__init__()
        self.loss_func = SmoothEdgeTermLossFunction(sigma_color, sigma_space, window_size, lp, w_smooth, w_edge, w_L2, isDetailEnhancement, isStylization)


    def forward(self, input, target):
        return self.loss_func(input, target)
