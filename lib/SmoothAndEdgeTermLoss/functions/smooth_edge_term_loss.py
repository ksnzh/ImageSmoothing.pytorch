import torch
from torch.autograd import Function
from .._ext import smooth_edge_term_loss


class SmoothEdgeTermLossFunction(Function):
    def __init__(ctx, sigma_color, sigma_space, window_size, lp, w_smooth, w_edge, w_L2, isDetailEnhancement, isStylization):
        ctx.sigma_color = - 1 / (sigma_color*sigma_color*2)
        ctx.sigma_space = - 1 / (sigma_space*sigma_space*2)
        ctx.window_size = window_size
        ctx.lp = lp
        ctx.w_smooth = w_smooth
        ctx.w_edge = w_edge
        ctx.w_L2 = w_L2
        ctx.isDetailEnhancement= isDetailEnhancement
        ctx.isStylization = isStylization

    def forward(ctx, input, target):
        assert input.is_cuda
        assert target.is_cuda

        ctx.input = input

        ctx.input_cnn = input[:, 0:3, :, :]
        ctx.input_edge = input[:, 3:4, :, :]
        ctx.target_yuv = target[:, 0:3, :, :]
        ctx.target_edge = target[:, 3:4, :, :]
        ctx.target_edge_mask = target[:, 4:5, :, :]

        ctx.output_smooth = torch.zeros_like(ctx.input_cnn)
        ctx.smooth_mask_pre = torch.zeros_like(ctx.input_edge)
        ctx.smooth_mask = torch.zeros_like(ctx.input_edge)
        ctx.w = torch.zeros_like(ctx.input_edge)

        smooth_edge_term_loss.smooth_edge_term_loss_forward_cuda(
            ctx.input_cnn,
            ctx.input_edge,
            ctx.target_yuv,
            ctx.target_edge,
            ctx.target_edge_mask,
            ctx.smooth_mask_pre,
            ctx.smooth_mask,
            ctx.w,
            ctx.output_smooth,
            ctx.sigma_color,
            ctx.sigma_space,
            ctx.window_size,
            ctx.lp,
            ctx.isDetailEnhancement,
            ctx.isStylization,
            ctx.w_L2
        )

        loss_edge = 0
        if torch.sum(ctx.target_edge_mask) != 0:
            sub = ctx.input_edge - ctx.target_edge
            ctx.gt = sub[ctx.target_edge_mask.byte()]
            loss_edge = torch.mean(torch.pow(ctx.gt, 2))

        loss_smooth = torch.mean(ctx.output_smooth)

        ctx.output = loss_edge * ctx.w_edge + loss_smooth * ctx.w_smooth

        return ctx.output

    def backward(ctx, grad_output):
        gradInput_smooth = torch.zeros_like(ctx.input_cnn)
        gradInput_edge = torch.zeros_like(ctx.input_edge)

        smooth_edge_term_loss.smooth_edge_term_loss_backward_cuda(
            ctx.input_cnn,
            ctx.smooth_mask,
            ctx.target_edge_mask,
            ctx.w,
            gradInput_smooth,
            ctx.sigma_color,
            ctx.window_size,
            ctx.lp,
            ctx.w_L2
        )

        if torch.sum(ctx.target_edge_mask) != 0:
            gradInput_edge[ctx.target_edge_mask.byte()] = 2 * ctx.gt / ctx.gt.shape[0]

        ctx.gradInput = torch.cat([
            ctx.w_smooth * gradInput_smooth / (3 * ctx.input_cnn.shape[2] * ctx.input_cnn.shape[3]),
            ctx.w_edge * gradInput_edge], dim=1)

        return grad_output * ctx.gradInput, None
