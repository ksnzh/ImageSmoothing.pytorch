#ifndef _SMOOTH_EDGE_TERM_LOSS_KERNEL
#define _SMOOTH_EDGE_TERM_LOSS_KERNEL

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifdef __cplusplus
extern "C" {
#endif

void smooth_edge_term_loss_forward_laucher(cudaStream_t stream, float* input_cnn, float* input_edge, float* target_yuv, float* target_edge, float* target_edge_mask, float* smooth_mask_pre, float* smooth_mask, float* weight, float* output, float sigma_color, float sigma_space, int window_size, float lp, int height, int width, int isDetailEnhancement, int isStylization, int w_L2);

void smooth_edge_term_loss_backward_laucher(cudaStream_t stream, float* input_cnn, float* smooth_mask, float* target_edge_mask, float* weight, float* gradInput, float sigma_color,int window_size, float lp, int height, int width, int w_L2);

#ifdef __cplusplus
}
#endif

#endif

