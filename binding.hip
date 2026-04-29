#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <assert.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

extern "C" __global__ void fast_hadamard_kernel(const float* in,float* out,const int N,const int step);
extern "C" __global__ void fast_hadamard_kernel_scale(const float* in,float* out,const float scale,const int N,const int step);

torch::Tensor fast_hadamard(torch::Tensor a,float s) {
    TORCH_CHECK(a.is_cuda(), "A must be GPU tensor");
    TORCH_CHECK(a.dim() >= 1, "A must have at least 1 dim");

    const auto type = a.scalar_type();
    TORCH_CHECK(
        type == torch::kFloat32 ||
        type == torch::kFloat16 ||
        type == torch::kBFloat16
    );

    int64_t outer = 1;
    for (int i = 0; i < a.dim() - 1; i++) outer *= a.size(i);
    const int N = a.size(-1);

    int N_padded = 1;
    while (N_padded < N) N_padded <<= 1;

    auto x = a.reshape({outer, N});

    auto padded = torch::zeros({outer, N_padded}, a.options());
    padded.narrow(1, 0, N).copy_(x);

    auto GPU_A = padded.to(torch::kFloat32);
    auto GPU_B = torch::zeros_like(GPU_A);

    float* in = GPU_A.data_ptr<float>();
    float* out = GPU_B.data_ptr<float>();

    int block_size = 32;
    int grid_size = ((outer * N_padded / 2) + block_size - 1) / block_size;
    dim3 block(block_size);
    dim3 grid(grid_size);

    int time = (int)round(log2(N_padded));

    for(int i = 0; i < time; i++) {
        if(i == time - 1) {
            fast_hadamard_kernel_scale<<<grid, block>>>(
                in, out, s,
                outer * N_padded, i
            );
            
        } else {
            fast_hadamard_kernel<<<grid, block>>>(
                in, out, outer * N_padded, i
            );
        }
        float* tmp = in;
        in = out;
        out = tmp;
    }

    HIP_ASSERT(hipDeviceSynchronize());

    auto res = (time % 2 == 0) ? GPU_A : GPU_B;

    res = res.narrow(1, 0, N);
    res = res.reshape(a.sizes());

    return res.to(a.scalar_type());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_hadamard", &fast_hadamard, "Fast hadamard for hip");
}
