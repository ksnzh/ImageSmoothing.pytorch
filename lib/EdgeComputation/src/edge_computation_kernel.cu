// #ifdef __cplusplus
// extern "C" {
// #endif

#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>
#include "edge_computation_kernel.h"



// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void EdgeComputationForward_kernel(const int num_kernels, float* input, float* output, int height, int width) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;

		int window_size = 1;
		for (int m = -window_size; m <= window_size; m++) {
			for (int n = -window_size; n <= window_size; n++) {
				if (y+m < 0 || y+m >= height || x+n < 0 || x+n >= width)
					continue;
				int image_offset = (y + m) * width + x + n;
				*(output + point_offset) += fabs(*(input + point_offset)-*(input + image_offset));
			}
		}

		if (y-2 >= 0)
			*(output + point_offset) += fabs(*(input + point_offset)-*(input + (y - 2) * width + x));
		if (y+2 < height)
			*(output + point_offset) += fabs(*(input + point_offset)-*(input + (y + 2) * width + x));
		if (x-2 >= 0)
			*(output + point_offset) += fabs(*(input + point_offset)-*(input + y * width + x - 2));
		if (x+2 < width)
			*(output + point_offset) += fabs(*(input + point_offset)-*(input + y * width + x + 2));

		*(output + point_offset) = *(output + point_offset)/6;
	}
}

void EdgeComputationForward(cudaStream_t stream, float* input, float* output, int height, int width)
{
	int dimSize = 1024;
	int num_kernels = height * width;
	int grid = (num_kernels + dimSize - 1) / dimSize;
	EdgeComputationForward_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, input, output, height, width);
}

__global__ void EdgeComputationBackward_kernel(const int num_kernels, float* input, float* gradOutput, float* gradInput, int height, int width) {

	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		int point_offset = index;
		int x = index % width;
		int y = index / width;

		int window_size = 1;
		for (int m = -window_size; m <= window_size; m++) {
			for (int n = -window_size; n <= window_size; n++) {
				if (y+m < 0 || y+m >= height || x+n < 0 || x+n >= width)
					continue;
				int image_offset = (y + m) * width + x + n;

				*(gradInput + point_offset) += (*(input + point_offset) > *(input + image_offset) ? 1 : -1) * *(gradOutput + point_offset);
				*(gradInput + point_offset) += (*(input + point_offset) > *(input + image_offset) ? 1 : -1) * *(gradOutput + image_offset);
			}
		}

		if (y-2 >= 0)
		{
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + (y - 2) * width + x) ? 1 : -1) * *(gradOutput + point_offset);
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + (y - 2) * width + x) ? 1 : -1) * *(gradOutput + (y - 2) * width + x);
		}
		if (y+2 < height)
		{
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + (y + 2) * width + x) ? 1 : -1) * *(gradOutput + point_offset);
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + (y + 2) * width + x) ? 1 : -1) * *(gradOutput + (y + 2) * width + x);
		}
		if (x-2 >= 0)
		{
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + y * width + x - 2) ? 1 : -1) * *(gradOutput + point_offset);
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + y * width + x - 2) ? 1 : -1) * *(gradOutput + y * width + x - 2);
		}
		if (x+2 < width)
		{
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + y * width + x + 2) ? 1 : -1) * *(gradOutput + point_offset);
			*(gradInput + point_offset) += (*(input + point_offset) > *(input + y * width + x + 2) ? 1 : -1) * *(gradOutput + y * width + x + 2);
		}

		*(gradInput + point_offset) = *(gradInput + point_offset)/6;

	}
}

void EdgeComputationBackward(cudaStream_t stream, float* input, float* gradOutput, float* gradInput, int height, int width)
{
	int dimSize = 1024;
	int num_kernels = height * width;
	int grid = (num_kernels + dimSize - 1) / dimSize;
	EdgeComputationBackward_kernel<<<grid, dimSize, 0, stream>>>(num_kernels, input, gradOutput, gradInput, height, width);
}


// #ifdef __cplusplus
// }
// #endif
