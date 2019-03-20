#include <stdio.h>
#include <stdint.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "support.h"
#include "lzkernel.cu"
#define BLOCK_SIZE 512

using namespace std;

int main(int argc, char** argv) {

    int s_b_s;
    int u_b_s;

    s_b_s = 8;
    u_b_s = 6;

    if(argc != 2) {
	cout << "Invalid number of arguments. Usage is: ./compressFile <filename>\n";
	return 0;
    }
    string fileName = argv[1];
    ifstream inFile(fileName.c_str());

    //Get the length of the file in characters
    unsigned int num_elements = 0;
    if(inFile) {
	inFile.seekg(0, inFile.end);
	num_elements = inFile.tellg();
	inFile.seekg(0, inFile.beg);
    }

    //Read file in character array
    char* h_input = new char[num_elements];
    inFile.read(h_input, num_elements);
    inFile.close();

    char* h_output = new char[2 * num_elements];

    printf("Allocating device variables...\n");
    char* d_input;
    char* d_output;

 //   unsigned int* d_bins;
    cudaError_t cuda_ret = cudaMalloc((void**)&d_input, num_elements * sizeof(char));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for character buffer\n");
    cudaMalloc((void**)&d_output, 2 * num_elements * sizeof(unsigned int));
  // if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for d_bins\n");

    printf("Copying host variables to device...\n");
    cuda_ret = cudaMemcpy(d_input, h_input, num_elements * sizeof(char), cudaMemcpyHostToDevice);
    //printCharArray(h_input, num_elements);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device\n");
    cuda_ret = cudaMemset(d_output, 0, 2 * num_elements * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory\n");

    cudaDeviceSynchronize();

    // Launch kernel ----------------------------------------------------------
   printf("Launching kernel..."); fflush(stdout);
  //  startTime(&timer);

    dim3 dimGrid((num_elements/BLOCK_SIZE),1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    lz77kernel<<<dimGrid, dimBlock>>>(d_input, d_output, s_b_s, u_b_s); 
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel\n");

    printf("Copying device variables to host...\n");
    cuda_ret = cudaMemcpy(h_output, d_output, 2 * num_elements * sizeof(int), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host\n");
    cudaDeviceSynchronize();

for (int n=0; n<2 * num_elements; ++n)
    cout << h_output[n] << ' ';
    cout << '\n';
    
   printf("Freeing memory...\n");
    delete[] h_input, h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}
