void edge_computation_forward_cuda(THCudaTensor * input, THCudaTensor * output);

void edge_computation_backward_cuda(THCudaTensor * input, THCudaTensor * gradOutput, THCudaTensor * gradInput);
