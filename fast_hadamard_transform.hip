#include <hip/hip_runtime.h>

extern "C" __global__ void fast_hadamard_kernel(const float* in,float* out,const int N,const int step) {
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if(i<N/2)
   {
        int step_pow = 1 << step; 

        int x_idx = (i / step_pow) * (step_pow * 2) + (i % step_pow);
        int y_idx = x_idx + step_pow;

        float a = in[x_idx];
        float b = in[y_idx];

        out[x_idx] = a + b;
        out[y_idx] = a - b;
   }
}

extern "C" __global__ void fast_hadamard_kernel_scale(const float* in,float* out,const float scale,const int N,const int step) {
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if(i<N/2)
   {
        int step_pow = 1 << step;

        int x_idx = (i / step_pow) * (step_pow * 2) + (i % step_pow);
        int y_idx = x_idx + step_pow;

        float a = in[x_idx];
        float b = in[y_idx];

        out[x_idx] = (a + b)*scale;
        out[y_idx] = (a - b)*scale;
   }
}

