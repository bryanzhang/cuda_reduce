#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace std;

static constexpr int narr = (1 << 24);
static constexpr int numThreads = 1024;
static constexpr int numBlocks = (narr + numThreads - 1) / numThreads / 8;
static constexpr int WARP_SIZE = 32;

__inline__ __device__ int warpReduce(int mySum) {
        mySum += __shfl_xor_sync(0xffffffff, mySum, 16);
        mySum += __shfl_xor_sync(0xffffffff, mySum, 8);
        mySum += __shfl_xor_sync(0xffffffff, mySum, 4);
        mySum += __shfl_xor_sync(0xffffffff, mySum, 2);
        mySum += __shfl_xor_sync(0xffffffff, mySum, 1);
        return mySum;
}

__global__ void reduceNeighbor(int* output, int* input) {
        __shared__ int smm[WARP_SIZE];

        int tid = threadIdx.x;
        int* inner_input = input + 8 * blockIdx.x * numThreads;
        int a0 = inner_input[tid];
        int a1 = inner_input[tid + numThreads];
        int a2 = inner_input[tid + 2 * numThreads];
        int a3 = inner_input[tid + 3 * numThreads];
        int a4 = inner_input[tid + 4 * numThreads];
        int a5 = inner_input[tid + 5 * numThreads];
        int a6 = inner_input[tid + 6 * numThreads];
        int a7 = inner_input[tid + 7 * numThreads];
        int mySum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
        mySum = warpReduce(mySum);
        int laneIdx = (tid % WARP_SIZE);
        int warpIdx = tid / WARP_SIZE;
        if (laneIdx == 0) {
                smm[warpIdx] = mySum;
        }
        __syncthreads();

        if (warpIdx == 0) {
                mySum = ( (tid < numThreads / WARP_SIZE) ? smm[laneIdx] : 0 );
                mySum = warpReduce(mySum);
                if (tid == 0) {
                        output[blockIdx.x] = mySum;
                }
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
        reduceNeighbor<<<grid, block>>>(d_sum, d_arr);
        // 检查并处理错误
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //      printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // }
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
