#ifndef _EDGE_COMPUTATION_KERNEL
#define _EDGE_COMPUTATION_KERNEL

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifdef __cplusplus
extern "C" {
#endif

void EdgeComputationForward(cudaStream_t stream, float* input, float* output, int height, int width);

void EdgeComputationBackward(cudaStream_t stream, float* input, float* gradOutput, float* gradInput, int height, int width);

#ifdef __cplusplus
}
#endif

#endif

