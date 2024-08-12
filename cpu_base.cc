#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace std;

int main() {
        srand(1);
        int size = (1 << 24);
        int* arr = new int[size];
        for (int i = 0; i < size; ++i) {
                arr[i] = (rand() & 0x7F);
        }


        // warmup
        for (int j = 0; j < 100; ++j) {
        int sum = 0;
        for (int i = 0; i < size; ++i) {
                sum += arr[i];
        }
        }

        int sum = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; ++i) {
                sum += arr[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cout << "cpu sum is: " << sum << endl << "elapsed time: " << elapsed.count() * 1000 << " ms." << endl;
        delete[] arr;
        return 0;
}
