


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <iomanip>
#include <string>
#include <cassert>
#include <cstring>
#include<sstream>

#include "main.h"
#include "sha256.cuh"

#define SHOW_INTERVAL_MS 2000
#define MAX_INTERMEDIATE_RESULTS 100
#define BYTES_PER_RESULT 48
#define ITERATIONS_PER_KERNEL 256
#define BLOCK_SIZE 256
#define NUMBLOCKS 163834u
#define NONCES_PER_KERNEL_EXEC (NUMBLOCKS * BLOCK_SIZE * ITERATIONS_PER_KERNEL)

static size_t difficulty = 1;
static size_t sampling_difficulty = 8;
int* g_found = nullptr;
int* d_resultsCount = nullptr;
static uint64_t nonce_low = 0;
static uint64_t nonce_high = 0;
static uint64_t user_nonce_low = 0;
static uint64_t user_nonce_high = 0;
static uint64_t last_nonce_since_update = 0;

// Last timestamp we printed debug info
static std::chrono::high_resolution_clock::time_point t_last_updated;

__device__ bool checkZeroPadding(unsigned char* sha, uint8_t difficulty) {
    uint8_t fullZeros = difficulty / 2;
    uint8_t remainder = difficulty % 2;

    // Check full zero bytes first
    for (uint8_t i = 0; i < fullZeros; i++) {
        if (sha[i] != 0) return false;
    }

    // Check the last byte based on remainder
    if (remainder) {
        if (sha[fullZeros] >= 0x10) return false;
    }

    return true;
}

__device__ void nonce_to_bytes(uint64_t nonce_low, uint64_t nonce_high, unsigned char* out) {
    out[0] = (unsigned char)(nonce_low >> 56);
    out[1] = (unsigned char)(nonce_low >> 48);
    out[2] = (unsigned char)(nonce_low >> 40);
    out[3] = (unsigned char)(nonce_low >> 32);
    out[4] = (unsigned char)(nonce_low >> 24);
    out[5] = (unsigned char)(nonce_low >> 16);
    out[6] = (unsigned char)(nonce_low >> 8);
    out[7] = (unsigned char)(nonce_low);
    out[8] = (unsigned char)(nonce_high >> 56);
    out[9] = (unsigned char)(nonce_high >> 48);
    out[10] = (unsigned char)(nonce_high >> 40);
    out[11] = (unsigned char)(nonce_high >> 32);
    out[12] = (unsigned char)(nonce_high >> 24);
    out[13] = (unsigned char)(nonce_high >> 16);
    out[14] = (unsigned char)(nonce_high >> 8);
    out[15] = (unsigned char)(nonce_high);
}

__constant__ unsigned char constant_bytes[4] = { 0xD8, 0x79, 0x9f, 0x50 };
__global__ void sha256_kernel(unsigned char* out_packed_results, int* out_num_results, int* out_found, const char* in_input_string, size_t in_input_string_size, uint8_t intermediate_difficulty, uint8_t difficulty, uint64_t nonce_offset_low, uint64_t nonce_offset_high) {
    __shared__ SHA256_CTX shared_ctx[BLOCK_SIZE];
    uint64_t nonce_low;
    uint64_t nonce_high;

    unsigned char sha[32];

    for (uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < NONCES_PER_KERNEL_EXEC;
        idx += blockDim.x * gridDim.x) {
        nonce_low = idx + nonce_offset_low;
        nonce_high = nonce_offset_high;

        if (nonce_low < idx) {
            nonce_high++;
        }

        sha256_init(&shared_ctx[threadIdx.x]);

        shared_ctx[threadIdx.x].data[0] = constant_bytes[0];
        shared_ctx[threadIdx.x].data[1] = constant_bytes[1];
        shared_ctx[threadIdx.x].data[2] = constant_bytes[2];
        shared_ctx[threadIdx.x].data[3] = constant_bytes[3];
        nonce_to_bytes(nonce_low, nonce_high, shared_ctx[threadIdx.x].data + 4);


        shared_ctx[threadIdx.x].datalen = 20;

        sha256_update(&shared_ctx[threadIdx.x], (unsigned char*)in_input_string, in_input_string_size);
        sha256_final(&shared_ctx[threadIdx.x], sha);

        sha256_init(&shared_ctx[threadIdx.x]);
        sha256_update(&shared_ctx[threadIdx.x], sha, 32);
        sha256_final(&shared_ctx[threadIdx.x], sha);

        if (checkZeroPadding(sha, intermediate_difficulty)) {
            int index = atomicAdd(out_num_results, 1);
            if (index < MAX_INTERMEDIATE_RESULTS) {
                unsigned char* target = out_packed_results + index * BYTES_PER_RESULT;
                memcpy(target, sha, 32);
                memcpy(target + 32, &nonce_low, 8);
                memcpy(target + 40, &nonce_high, 8);
                // Check for primary difficulty
                if (checkZeroPadding(sha, difficulty)) {
                    atomicExch(out_found, 1);
                }
            }
        }
    }
}

void pre_sha256() {
    checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

void print_hex(char* someHexData, size_t length) {
    for (size_t i = 0; i < length; i++) {
        printf("%02x", someHexData[i]);
    }
}

void print_hex(unsigned char* someHexData, size_t length) {
    for (size_t i = 0; i < length; i++) {
        printf("%02x", someHexData[i]);
    }
}

void print_state() {
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> last_show_interval = t2 - t_last_updated;

    if (last_show_interval.count() > SHOW_INTERVAL_MS) {
        std::chrono::duration<double, std::milli> span = t2 - t_last_updated;
        float ratio = span.count() / 1000;

        std::cout << std::fixed << static_cast<uint64_t>((nonce_low - last_nonce_since_update) / ratio) << " hash(es)/s" << std::endl;


        std::cout << std::fixed << "Nonce : " << nonce_low << std::endl;

        t_last_updated = std::chrono::high_resolution_clock::now();
        last_nonce_since_update = nonce_low;
    }
}

void hex_to_u64s(const std::string& hex, uint64_t& high, uint64_t& low) {

    std::stringstream ss;

    ss << std::hex << hex.substr(0, 16); // First 16 characters
    ss >> high;
    ss.clear();
    ss << std::hex << hex.substr(16, 16); // Last 16 characters
    ss >> low;
}

void hex_string_to_bytes(const char* hex_input, char* byte_output, size_t input_size) {
    for (size_t i = 0; i < input_size; i += 2) {
        // Take two characters from the hex string at a time
        char hex_byte[3];
        hex_byte[0] = hex_input[i];
        hex_byte[1] = hex_input[i + 1];
        hex_byte[2] = '\0';  // Null-terminate the string

        // Convert the two-character hex string to a byte
        byte_output[i / 2] = (char)strtol(hex_byte, NULL, 16);
    }
}

void to_hex_string(uint64_t a, uint64_t b, char* output) {
    unsigned char bytes[16];

    // Extract bytes
    for (int i = 0; i < 8; i++) {
        bytes[i] = (a >> (8 * (7 - i))) & 0xFF;
        bytes[8 + i] = (b >> (8 * (7 - i))) & 0xFF;
    }

    // Convert bytes to hex
    for (int i = 0; i < 16; i++) {
        sprintf(output + 2 * i, "%02x", bytes[i]);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: <program> <message> <nonce hex> <difficulty> <sampling_difficulty>" << std::endl;
        return 1;
    }

    cudaSetDevice(0);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    t_last_updated = std::chrono::high_resolution_clock::now();

    std::string in, user_nonce_hex;

    in = argv[1];
    user_nonce_hex = argv[2];

    hex_to_u64s(user_nonce_hex, user_nonce_low, user_nonce_high);

    difficulty = std::stoul(argv[3]);
    sampling_difficulty = std::stoul(argv[4]);

    // hex decode user input "string"
    size_t byte_array_size = in.size() / 2;
    char* byte_array = new char[byte_array_size];
    hex_string_to_bytes(in.c_str(), byte_array, in.size());

    // Input string for the device
    char* d_in = nullptr;

    // Create the input string for the device
    cudaMalloc(&d_in, byte_array_size);
    cudaMemcpy(d_in, byte_array, byte_array_size, cudaMemcpyHostToDevice);


    cudaMallocManaged(&g_found, sizeof(int));
    *g_found = 0;


    nonce_low += user_nonce_low;
    nonce_high += user_nonce_high;


    pre_sha256();


    unsigned char* d_packedResults;


    cudaMallocManaged(&d_packedResults, MAX_INTERMEDIATE_RESULTS * BYTES_PER_RESULT);
    cudaMallocManaged(&d_resultsCount, sizeof(int));
    *d_resultsCount = 0;


    while (!*g_found) {
        sha256_kernel << < NUMBLOCKS, BLOCK_SIZE >> > (d_packedResults, d_resultsCount, g_found, d_in, byte_array_size, sampling_difficulty, difficulty, nonce_low, nonce_high);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("Device error");
        }
        nonce_low += NONCES_PER_KERNEL_EXEC;
        if (nonce_low < NONCES_PER_KERNEL_EXEC) {
            nonce_high++;
        }

        char hex_output[33];
        for (int i = 0; i < *d_resultsCount; i++) {
            int offset = i * BYTES_PER_RESULT;

            unsigned char* currentSHA = d_packedResults + offset;
            uint64_t* currentNonceLow = (uint64_t*)(d_packedResults + offset + 32);
            uint64_t* currentNonceHigh = (uint64_t*)(d_packedResults + offset + 40);

            print_hex(currentSHA, 32);
            printf("|");
            to_hex_string(*currentNonceLow, *currentNonceHigh, hex_output);
            printf("%s", hex_output);

            printf("\n"); // New line for each result for readability.
            fflush(stdout);
        }

        *d_resultsCount = 0;

        //print_state();
    }


    cudaFree(g_found);
    cudaFree(d_packedResults);
    cudaFree(d_resultsCount);
    cudaFree(d_in);
    delete[] byte_array;
    cudaDeviceReset();

    return 0;
}