void smooth_edge_term_loss_forward_cuda(THCudaTensor *input_cnn, THCudaTensor *input_edge, THCudaTensor *target_yuv, THCudaTensor *target_edge, THCudaTensor *target_edge_mask, THCudaTensor *smooth_mask_pre, THCudaTensor *smooth_mask, THCudaTensor *weight, THCudaTensor *output, float sigma_color, float sigma_space, int window_size, float lp, int isDetailEnhancement, int isStylization, int w_L2);

void smooth_edge_term_loss_backward_cuda(THCudaTensor *input_cnn, THCudaTensor *smooth_mask, THCudaTensor *target_edge_mask, THCudaTensor *weight, THCudaTensor *gradInput, float sigma_color, int window_size, float lp, int w_L2);