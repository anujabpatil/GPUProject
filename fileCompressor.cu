#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "support.h"
#include "kernel.cu"
#include "huffmanNode.h"
#include "huffmanUtil.h"

//extern const int NUM_BINS;

using namespace std;

int main(int argc, char** argv) {
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

    unsigned int* h_bins = new unsigned int[NUM_BINS];

    printf("Allocating device variables...\n");
    char* d_input;
    unsigned int* d_bins;
    cudaError_t cuda_ret = cudaMalloc((void**)&d_input, num_elements * sizeof(char));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for character buffer\n");
    cuda_ret = cudaMalloc((void**)&d_bins, NUM_BINS * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for d_bins\n");

    printf("Copying host variables to device...\n");
    cuda_ret = cudaMemcpy(d_input, h_input, num_elements * sizeof(char), cudaMemcpyHostToDevice);
    //printCharArray(h_input, num_elements);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device\n");
    cuda_ret = cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory\n");

    cudaDeviceSynchronize();

    printf("Launching kernel...\n");
    histogram(d_input, d_bins, num_elements);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel\n");

    printf("Copying device variables to host...\n");
    cuda_ret = cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host\n");
    cudaDeviceSynchronize();
   
    /*for(int i=0; i<NUM_BINS; i++) {
	if(i>=0 && i <= 25) printf("%c frequency is %d \n", i + 97, h_bins[i]);
	else if(i>= 26 && i <= 35) printf("%c frequency is %d \n", i + 22, h_bins[i]);
	else if(i == 36) printf("\" \" frequency is %d \n", h_bins[i]);
	else if(i == 37) printf(". frequency is %d \n", h_bins[i]);
 	else if(i == 38) printf(", frequency is %d \n", h_bins[i]);
	else if(i == 39) printf("\\n frequency is %d \n", h_bins[i]);
    }*/

    //Build encoding tree
    HuffmanNode* encodingTree = buildEncodingTree(h_bins);

    //Build encoding array
    string* codeArr = buildEncodingArray(encodingTree);

    /*for(int i = 0; i < NUM_BINS; i++) {
	cout << "Code for " << i << ": " << codeArr[i] << endl; 
    }*/

    //Find maximum length of codes
    int maxLength = getMaxCodeLength(codeArr);

    //Convert string* to char**
    char** characterCodes = new char*[NUM_BINS];
    for(int i = 0; i < NUM_BINS; i++) {
       char* code = (char*) codeArr[i].c_str();
    }   
    
    printf("Freeing memory...\n");
    delete[] h_input, h_bins;
    cudaFree(d_input);
    cudaFree(d_bins);
}

