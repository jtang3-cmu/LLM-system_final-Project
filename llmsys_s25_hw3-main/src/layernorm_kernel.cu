#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/



template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN3_2

  // Get thread and block indices
  int tid = threadIdx.x; // Thread ID within the block
  int token_id = blockIdx.x; // Each block processes a different token (batch element)

  // Compute the number of float4 elements in the hidden size
  int float4_count = hidden_size;  
  int total_elems = float4_count * 4; // Total elements considering float4 vectorization
  
  // Reinterpret input, scale, bias, and output tensors as float4 arrays for efficient memory access
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + token_id * float4_count;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  float4 *out_f4 = reinterpret_cast<float4 *>(ln_res) + token_id * float4_count;
  
  // Local accumulators for sum and squared sum (to compute mean and variance)
  float local_sum = 0.f;
  float local_sum2 = 0.f;

  // Compute partial sum of mean and variance across hidden size
  for (int idx = tid; idx < float4_count; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    local_sum += (val.x + val.y + val.z + val.w);
    local_sum2 += (val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w);
  }

  // Perform block-wide reduction to compute total sum and squared sum
  local_sum = blockReduceSum<float>(local_sum);
  local_sum2 = blockReduceSum<float>(local_sum2);

  // Only thread 0 computes mean and variance and stores them
  if (tid == 0) {
    float mean = local_sum / (float)total_elems; // Compute mean
    float var = (local_sum2 / (float)total_elems) - mean * mean; // Compute variance
    var += LN_EPSILON; // Add epsilon for numerical stability
    means[token_id] = mean; // Store mean for the token
    vars[token_id] = var; // Store variance for the token
  }
  __syncthreads(); // Synchronize threads to ensure mean and variance are stored before proceeding

  // Load computed mean and variance
  float mean_val = means[token_id]; 
  float variance_val = vars[token_id]; 
  float inv_std = rsqrtf(variance_val);

  // Compute Layer Normalization output
  for (int idx = tid; idx < float4_count; idx += blockDim.x) {
    float4 x4 = inp_f4[idx];
    
    // Normalize input using mean and standard deviation
    x4.x = (x4.x - mean_val) * inv_std;
    x4.y = (x4.y - mean_val) * inv_std;
    x4.z = (x4.z - mean_val) * inv_std;
    x4.w = (x4.w - mean_val) * inv_std;

    // Load scale and bias parameters
    float4 scale4 = scale_f4[idx];
    float4 bias4 = bias_f4[idx];

    // Apply scale and bias transformation
    x4.x = x4.x * scale4.x + bias4.x;
    x4.y = x4.y * scale4.y + bias4.y;
    x4.z = x4.z * scale4.z + bias4.z;
    x4.w = x4.w * scale4.w + bias4.w;

    // Store the output
    out_f4[idx] = x4;
  }

  /// END ASSIGN3_2
}


extern "C" {
  void launch_layernorm(float *ln_res, float *vars, float *means,
                                const float *inp, const float *scale,
                                const float *bias, int batch_size, int hidden_dim,
                                cudaStream_t stream) {
    if (hidden_dim % 4 != 0) {
      throw std::runtime_error("violate hidden_dim % 4 = 0");
    }
    int float_size = sizeof(float);
    int input_size = batch_size * hidden_dim * float_size;
    int scale_size = hidden_dim * float_size;
    int bias_size = hidden_dim * float_size;
    int output_size = batch_size * hidden_dim * float_size;
    int mean_size = batch_size * float_size;
    int var_size = batch_size * float_size;
  
  
    float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
    cudaMalloc((void **)&d_ln_res, output_size);
    cudaMalloc((void **)&d_vars, var_size);
    cudaMalloc((void **)&d_means, mean_size);
    cudaMalloc((void **)&d_inp, input_size);
    cudaMalloc((void **)&d_scale, scale_size);
    cudaMalloc((void **)&d_bias, bias_size);
  
    cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);
  
    // For using float4
    hidden_dim >>= 2;
    int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
    dim3 grid_dim(batch_size);
    dim3 block_dim(nthread);
  
    ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
        d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);
  
    // Copy back to the host
    cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  
    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }
  
    // Free memory on device
    cudaFree(d_ln_res);
    cudaFree(d_vars);
    cudaFree(d_means);
    cudaFree(d_inp);
    cudaFree(d_scale);
    cudaFree(d_bias);
  
  }
  }


/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  // cg::thread_block b = cg::this_thread_block();
  // cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Get thread and block indices
  int j = blockIdx.x * TILE_DIM + threadIdx.x; // Feature dimension index
  if (j >= width) return; // Ensure the thread is within valid width range

  float local_dgamma = 0.f; // Local gradient accumulation for gamma
  float local_dbeta = 0.f; // Local gradient accumulation for beta

  // Each thread iterates through all rows (tokens) in the batch
  for (int i = threadIdx.y; i < rows; i += blockDim.y) {
    int idx = i * width + j; // Compute index in the flattened input tensor
    float mu = means[i]; // Retrieve mean for the current token
    float inv_std = rsqrtf(vars[i]); // Compute inverse standard deviation
    float xhat = (inp[idx] - mu) * inv_std; // Normalize input value
    float grad = out_grad[idx]; // Gradient from output
    // Compute partial derivatives
    local_dgamma += grad * xhat;
    local_dbeta += grad;
  }
  // Shared memory for intra-block reduction
  __shared__ float s_gamma[TILE_DIM][TILE_DIM];
  __shared__ float s_beta[TILE_DIM][TILE_DIM];

  // Store partial sums into shared memory
  s_gamma[threadIdx.y][threadIdx.x] = local_dgamma;
  s_beta[threadIdx.y][threadIdx.x] = local_dbeta;
  __syncthreads();

  // Perform intra-block reduction along the y-dimension
  for (int stride = blockDim.y / 2; stride > 0; stride /= 2) {
    if (threadIdx.y < stride) {
      s_gamma[threadIdx.y][threadIdx.x] += s_gamma[threadIdx.y + stride][threadIdx.x];
      s_beta[threadIdx.y][threadIdx.x] += s_beta[threadIdx.y + stride][threadIdx.x];
    }
    __syncthreads();
  }
  // The first row of threads accumulates the final sum to global memory
  if (threadIdx.y == 0) {
    atomicAdd(&gamma_grad[j], s_gamma[0][threadIdx.x]); // Accumulate gamma gradient
    atomicAdd(&betta_grad[j], s_beta[0][threadIdx.x]); // Accumulate beta gradient
  }
  // END ASSIGN3_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int float4_count) {
  int i = blockIdx.x; // Each block processes a different sample (batch element)
  int tid = threadIdx.x; // Thread index within the block
  int D = float4_count * 4; // Total number of elements in the feature dimension

  // Reinterpret input, gradient, and output tensors as float4 arrays for efficient memory access
  const float4 *inp_f4 = reinterpret_cast<const float4*>(inp) + i * float4_count;
  const float4 *out_grad_f4 = reinterpret_cast<const float4*>(out_grad) + i * float4_count;

  // Local accumulators for gradient computation
  float local_dxhat_sum = 0.f;
  float local_dxhat_xhat_sum = 0.f;
  float inv_std = rsqrtf(vars[i]); // Compute inverse standard deviation

  // Step 1: Compute intermediate values for gradient normalization
  for (int j = tid; j < float4_count; j += blockDim.x) {
    int base = j * 4;
    float4 in_val = inp_f4[j]; // Load input value
    float4 out_val = out_grad_f4[j]; // Load output gradient
    for (int k = 0; k < 4; k++) {
      float x = ((float*)&in_val)[k]; // Extract individual elements from float4
      float dy = ((float*)&out_val)[k]; // Extract corresponding gradient
      float xhat = (x - means[i]) * inv_std; // Compute normalized input
      float dxhat = dy * gamma[base + k]; // Compute scaled gradient
      local_dxhat_sum += dxhat; // Accumulate sum of dxhat
      local_dxhat_xhat_sum += dxhat * xhat; // Accumulate sum of dxhat * xhat
    }
  }

  // Perform block-wide reduction for dxhat and dxhat_xhat
  float sum_dxhat = blockReduceSum<float>(local_dxhat_sum);
  float sum_dxhat_xhat = blockReduceSum<float>(local_dxhat_xhat_sum);
  __syncthreads();
  
  // Step 2: Compute input gradient using the computed sums
  for (int j = tid; j < float4_count; j += blockDim.x) {
    int base = j * 4;
    float4 in_val = inp_f4[j]; // Load input
    float4 out_val = out_grad_f4[j]; // Load output gradient
    float4 grad_result; // Output gradient storage
    for (int k = 0; k < 4; k++) {
      float x = ((float*)&in_val)[k]; // Extract input value
      float dy = ((float*)&out_val)[k]; // Extract output gradient
      float xhat = (x - means[i]) * inv_std; // Compute normalized input
      float dxhat = dy * gamma[base + k]; // Compute scaled gradient
      float dx = inv_std * (dxhat - (sum_dxhat + xhat * sum_dxhat_xhat) / ((float)D)); // Compute final input gradient using layer norm backward formula
      ((float*)&grad_result)[k] = dx; // Store computed gradient
    }
    reinterpret_cast<float4*>(inp_grad)[i * float4_count + j] = grad_result; // Write computed gradient back to global memory
  }
}



extern "C" {
  void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                           const float *out_grad, const float *inp, const float *gamma,
                           const float *betta, const float *vars,
                           const float *means, int batch_size, int hidden_dim,
                           cudaStream_t stream_1, cudaStream_t stream_2) {
    
    // Allocate device memory
    float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
    int grad_output_size = batch_size * hidden_dim * sizeof(float);
    int gamma_betta_size = hidden_dim * sizeof(float);
    int vars_means_size = batch_size * sizeof(float);
  
    cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
    cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
    cudaMalloc((void **)&d_inp_grad, grad_output_size);
    cudaMalloc((void **)&d_out_grad, grad_output_size);
    cudaMalloc((void **)&d_inp, grad_output_size);
    cudaMalloc((void **)&d_gamma, gamma_betta_size);
    cudaMalloc((void **)&d_betta, gamma_betta_size);
    cudaMalloc((void **)&d_vars, vars_means_size);
    cudaMalloc((void **)&d_means, vars_means_size);
  
    // Copy memory to device
    cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);
  
    // Launch kernels
    // Compute grad of gamma and betta
    // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
    // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
    dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);
    ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
        d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
        d_means, batch_size, hidden_dim);
  
    // Compute grad of input
    if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
      throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
    }
    hidden_dim >>= 2;
    int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
    ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
        d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);
  
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  
    // Copy back to host
    cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);
  
    // Free device memory
    cudaFree(d_gamma_grad);
    cudaFree(d_betta_grad);
    cudaFree(d_inp_grad);
    cudaFree((void *)d_out_grad);
    cudaFree((void *)d_inp);
    cudaFree((void *)d_gamma);
    cudaFree((void *)d_betta);
    cudaFree((void *)d_vars);
    cudaFree((void *)d_means);
  }}
  }}