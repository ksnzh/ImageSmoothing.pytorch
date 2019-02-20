#ifndef _EDGE_DETECTOR_KERNEL
#define _EDGE_DETECTOR_KERNEL

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define PI 3.1415926535897932384626433832

#ifdef __cplusplus
extern "C" {
#endif

void EdgeDetector(cudaStream_t stream, float* input_image, float* input_edge, float* output_preserve, float* output_eliminate, int height, int width, int isSmoothing);

#ifdef __cplusplus
}
#endif

#endif

