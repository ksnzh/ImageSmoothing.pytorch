#include <THC/THC.h>
#include <math.h>
#include "edge_computation_kernel.h"

extern THCState *state;

void edge_computation_forward_cuda(THCudaTensor * input, THCudaTensor * output)
{
    int batchSize = THCudaTensor_size(state, input, 0);
	int plane = THCudaTensor_size(state, input, 1);
    int height = THCudaTensor_size(state, input, 2);
    int width = THCudaTensor_size(state, input, 3);

	THCudaTensor *input_n = THCudaTensor_new(state);
	THCudaTensor *output_n = THCudaTensor_new(state);

	// For each elt in batch, do:
	for (int elt = 0; elt < batchSize; elt ++) {
		// Matrix mulitply per output:
		THCudaTensor_select(state, input_n, input, 0, elt);
		THCudaTensor_select(state, output_n, output, 0, elt);

		EdgeComputationForward(THCState_getCurrentStream(state),
				THCudaTensor_data(state, input_n),
				THCudaTensor_data(state, output_n),
				height, width);
	}

	THCudaTensor_free(state, input_n);
	THCudaTensor_free(state, output_n);
}

void edge_computation_backward_cuda(THCudaTensor * input, THCudaTensor * gradOutput, THCudaTensor * gradInput)
{
    int batchSize = THCudaTensor_size(state, input, 0);
	int plane = THCudaTensor_size(state, input, 1);
    int height = THCudaTensor_size(state, input, 2);
    int width = THCudaTensor_size(state, input, 3);

	THCudaTensor *input_n = THCudaTensor_new(state);
	THCudaTensor *gradOutput_n = THCudaTensor_new(state);
	THCudaTensor *gradInput_n = THCudaTensor_new(state);

	// For each elt in batch, do:
	for (int elt = 0; elt < batchSize; elt ++) {
		// Matrix mulitply per output:
		THCudaTensor_select(state, input_n, input, 0, elt);
		THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);
		THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);

		EdgeComputationBackward(THCState_getCurrentStream(state),
				THCudaTensor_data(state, input_n),
				THCudaTensor_data(state, gradOutput_n),
				THCudaTensor_data(state, gradInput_n),
				height, width);
	}

	THCudaTensor_free(state, input_n);
	THCudaTensor_free(state, gradOutput_n);
	THCudaTensor_free(state, gradInput_n);
}
