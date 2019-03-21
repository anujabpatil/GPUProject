#include <stdio.h>
#include <stdint.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "lzkernel.cu"
#include "support.h"
#define BLOCK_SIZE 512

using namespace std;

int main(int argc, char** argv) {

    int s_b_s;
    int u_b_s;
	Timer timer;
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

    int num = (num_elements/BLOCK_SIZE);

    char* h_output = new char[2 * num * BLOCK_SIZE];

    printf("Allocating device variables...\n");
    char* d_input;
    char* d_output;

 //   unsigned int* d_bins;
    cudaError_t cuda_ret = cudaMalloc((void**)&d_input, num_elements * sizeof(char));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for character buffer\n");
    cudaMalloc((void**)&d_output,2 * num * BLOCK_SIZE * sizeof(unsigned int));
  // if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for d_bins\n");

    printf("Copying host variables to device...\n");
    cuda_ret = cudaMemcpy(d_input, h_input, num_elements * sizeof(char), cudaMemcpyHostToDevice);
    //printCharArray(h_input, num_elements);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device\n");
    cuda_ret = cudaMemset(d_output, 0, 2 * num * BLOCK_SIZE * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory\n");

    cudaDeviceSynchronize();

    // Launch kernel ----------------------------------------------------------
   printf("Launching kernel..."); fflush(stdout);
 startTime(&timer);

    dim3 dimGrid((num_elements/BLOCK_SIZE),1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    lz77kernel<<<dimGrid, dimBlock>>>(d_input, d_output, s_b_s, u_b_s); 
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel\n");
stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Copying device variables to host...\n");
    cuda_ret = cudaMemcpy(h_output, d_output,2 * num * BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host\n");
    cudaDeviceSynchronize();
 int n ;
for ( n=0; n < (2 * BLOCK_SIZE * num); ++n)
    cout << h_output[n] << ' ';
    cout << n;
    cout << '\n';
   int count = 0;
 	int r ;
for ( r = 0; r < (2 * BLOCK_SIZE * num); r = r + 2)
{
//	count++;
//	printf("count = %c\n",h_output[r]);
	if ( h_output[r] != '1')
	{
		int x = h_output[ r + 1] - '0';
		r = 2*(x-1) + r;
	}
	else if ( h_output[r] == 'A')
	{
		count = count - 2;
	}
	count = count + 2;
}
printf("count = %d\n",count);
 
   printf("Freeing memory...\n");
  // free(h_input);
   // free(h_output);
  //  free(C_h);
 delete[] h_input, h_output;
   cudaFree(d_input);
    cudaFree(d_output);
}
