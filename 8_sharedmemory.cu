#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace std;

static constexpr int narr = (1 << 24);
static constexpr int numThreads = 1024;
static constexpr int numBlocks = (narr + numThreads - 1) / numThreads / 8;

template <unsigned int iBlockSize>
__global__ void reduceNeighbor(int* output, int* input) {
        __shared__ int smm[iBlockSize];

        int tid = threadIdx.x;
        int* inner_input = input + 8 * blockIdx.x * iBlockSize;
        smm[tid] = inner_input[tid];
        smm[tid] += inner_input[tid + iBlockSize];
        smm[tid] += inner_input[tid + 2 * iBlockSize];
        smm[tid] += inner_input[tid + 3 * iBlockSize];
        smm[tid] += inner_input[tid + 4 * iBlockSize];
        smm[tid] += inner_input[tid + 5 * iBlockSize];
        smm[tid] += inner_input[tid + 6 * iBlockSize];
        smm[tid] += inner_input[tid + 7 * iBlockSize];
        __syncthreads();

        if (iBlockSize >= 1024 && tid < 512) {
                smm[tid] += smm[tid + 512];
                __syncthreads();
        }

        if (iBlockSize >= 512 && tid < 256) {
                smm[tid] += smm[tid + 256];
                __syncthreads();
        }

        if (iBlockSize >= 256 && tid < 128) {
                smm[tid] += smm[tid + 128];
                __syncthreads();
        }

        if (iBlockSize >= 128 && tid < 64) {
                smm[tid] += smm[tid + 64];
                __syncthreads();
        }

        // unrolling warp
        if (tid < 32) {
                volatile int* vmem = smm;  // NOTE: very important 'volatile'
                vmem[tid] += vmem[tid + 32];
                vmem[tid] += vmem[tid + 16];
                vmem[tid] += vmem[tid + 8];
                vmem[tid] += vmem[tid + 4];
                vmem[tid] += vmem[tid + 2];
                vmem[tid] += vmem[tid + 1];
        }

        if (tid == 0) {
                output[blockIdx.x] = smm[0];
        }
}


struct TestResult {
        int sum;
        double elapsed;  // us
};

TestResult test(int* d_arr, int* h_arr, int* d_sum, int* h_sum) {
        cudaMemcpy(d_arr, h_arr, narr * sizeof(int), cudaMemcpyHostToDevice);
        dim3 grid(numBlocks);
        dim3 block(numThreads);
        auto start = std::chrono::high_resolution_clock::now();
        reduceNeighbor<numThreads><<<grid, block>>>(d_sum, d_arr);
        cudaDeviceSynchronize();
        cudaMemcpy(h_sum, d_sum, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
        int sum = 0;
        for (int i = 0; i < numBlocks; ++i) {
                sum += h_sum[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        TestResult ret = { sum, elapsed.count() };
        return ret;
}

int main() {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0); // 0是设备ID
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

        srand(1);
        int* h_arr = new int[narr];
        for (int i = 0; i < narr; ++i) {
                h_arr[i] = (rand() & 0x7f);
        }

        int* d_arr = nullptr;
        cudaMalloc(&d_arr, narr * sizeof(int));
        int* h_sum = new int[numBlocks];
        int* d_sum = nullptr;
        cudaMalloc(&d_sum, numBlocks * sizeof(int));

        // warmup
        for (int i = 0; i < 100; ++i) {
                test(d_arr, h_arr, d_sum, h_sum);
        }

        // test
        TestResult result = test(d_arr, h_arr, d_sum, h_sum);
        cout << "gpu sum is: " << result.sum << endl << "elapsed time: " << result.elapsed * 1000 << " ms." << endl;

        delete[] h_arr;
        delete[] h_sum;
        cudaFree(d_arr);
        cudaFree(d_sum);
        return 0;
}
