/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to
#define BLOCK_SIZE 512
#define NUM_BINS 37
#define SPACE_INDEX 36
// INSERT KERNEL(S) HERE

__global__ void fillBins(unsigned char* input, unsigned int* bins, unsigned int num_elements)
{
    
   __shared__ unsigned int private_histogram[NUM_BINS];

    //Initializing private histogram bins
//    int bin_stride = 0;

//    while((threadIdx.x + bin_stride) < num_bins) {
//	private_histogram[threadIdx.x + bin_stride] = 0;
//	bin_stride += blockDim.x;
//    }
//    __syncthreads();

    //Computation of private histogram
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

 if (i < NUM_BINS)
	{
	 private_histogram[i] = 0;
	}
__syncthreads():

 /*   while(i < num_elements) {
     	atomicAdd(&private_histogram[input[i]], 1);
	i += stride;
    }
    __syncthreads();*/
while(i < num_elements) {
if(input[i] >= 65 && input[i] <= 122)
{
	 atomicAdd(&private_histogram[input[i]-97], 1);
}
else if (input[i] >= 48 && input[i] <= 57)
{
	 atomicAdd(&private_histogram[input[i]-22], 1);
}
else if (input[i] == 32)
{
	 atomicAdd(&private_histogram[SPACE_INDEX], 1);
}
i += stride;
}
__syncthreads():
    //Merging private history bins with global history bins
 /*   bin_stride = 0;
    while((threadIdx.x + bin_stride) < num_bins) {
        atomicAdd(&bins[threadIdx.x + bin_stride], private_histogram[threadIdx.x + bin_stride]);
        bin_stride += blockDim.x;
    }
    __syncthreads();*/
	if (i < NUM_BINS)
        {
         	 atomicAdd(&bins[i], private_histogram[i]);
        }
	 __syncthreads();
}





/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned char* input, unsigned int* bins, unsigned int num_elements) 
{
    // INSERT CODE HERE
    dim3 dimGrid((num_elements - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    fillBins<<<dimGrid, dimBlock>>>(input, bins, num_elements);
}
