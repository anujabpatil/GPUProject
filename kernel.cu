/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

#include <cstdio>

#define BLOCK_SIZE 512
#define NUM_BINS 40
#define SPACE_INDEX 36
#define PERIOD_INDEX 37
#define COMMA_INDEX 38
#define NEW_LINE 39

// INSERT KERNEL(S) HERE

/*
 * This function calculates the frequency of each character in the character buffer. It does so by creating local
 * histograms and finally merging them with global histogram.
 */
__global__ void calculateFrequency(char* input, unsigned int* bins, unsigned int num_elements)
{
    
   __shared__ unsigned int private_histogram[NUM_BINS];

    //Computation of private histogram
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (threadIdx.x < NUM_BINS) {
	private_histogram[threadIdx.x] = 0;
    }
    __syncthreads();
    
    while(i < num_elements) {
        if(input[i] >= 97 && input[i] <= 122) { //if the character is small alphabet from a to z
	    atomicAdd(&private_histogram[input[i]-97], 1);
        } else if(input[i] >= 48 && input[i] <= 57) { //if the character is a digit from 0 to 9
	    atomicAdd(&private_histogram[input[i]-22], 1);
        } else if(input[i] == 32) { //if the character is a space
	    atomicAdd(&private_histogram[SPACE_INDEX], 1);
        } else if(input[i] == 46) { //if the character is a fullstop 
	    atomicAdd(&private_histogram[PERIOD_INDEX], 1);
	} else if(input[i] == 44) { //if the character is a comma
	    atomicAdd(&private_histogram[COMMA_INDEX], 1);
	} else if(input[i] == 10) { //if the character is a new line character
	    atomicAdd(&private_histogram[NEW_LINE], 1);
	}
        i += stride;
    }
    __syncthreads();
	
    if (threadIdx.x < NUM_BINS) {
      	 atomicAdd(&bins[threadIdx.x], private_histogram[threadIdx.x]);
    }
    __syncthreads();
}





/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(char* input, unsigned int* bins, unsigned int num_elements) 
{
    // INSERT CODE HERE
    dim3 dimGrid((num_elements - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    calculateFrequency<<<dimGrid, dimBlock>>>(input, bins, num_elements);
}
