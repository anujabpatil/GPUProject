#define BLOCK_SIZE 512
/*int max(int x, int y) 
{ 
  int retvalue = (x > y) ? x : y;
  return retvalue; 
}*/


__global__ void lz77kernel(char *in_d, char *out_d, int search_buffer_size, int uncoded_buffer_size) 
{
//	int maximum = 0;
        int i = threadIdx.x + blockIdx.x * blockDim.x;

	int k;
	int j;
//	int l;
	char length;
	char offset;
	 //int maximum;

//	int total_size = search_buffer_size + uncoded_buffer_size;
//	int shared_data_size = (total_size-1) + BLOCK_SIZE;
	__shared__ char data[BLOCK_SIZE];

	data[threadIdx.x] = in_d[i];
	
	__syncthreads();

if(threadIdx.x < 499)
{
	int search_buffer_first = threadIdx.x;
        int uncoded_first = threadIdx.x + search_buffer_size;

        for ( j = search_buffer_first; j < search_buffer_first + search_buffer_size; j++)
	{
		if( data[j] == data[uncoded_first])
		{
			 int last_searchbuffer_match = j;
			for(k = (uncoded_first); k < (uncoded_first + uncoded_buffer_size) ; k++)
			{	
			//	int last_searchbuffer_match = j;		
				if(data[k] == data[last_searchbuffer_match])
				{
					length++;
					last_searchbuffer_match++;
				}
				else
				{	
					last_searchbuffer_match = 0;	
					break;	
				}
			}
		}
		if (length >= 3)
		{	
			offset = (uncoded_first - j);	
			break;
			
		}
		else
		{
			length = 0;	
		}
	
	}
 
	if ( length == 0)
	{
	//	char a = 'A';
		int x = 1;
		out_d[i+i] = x + '0';
		out_d[i + i + 1] = data[uncoded_first];
	}
	else
	{
		char a = offset + '0' ; 
		char b = length + '0';
		out_d[i+i] = a;
          	out_d[i + i + 1] = b;
	}
}
else
{
	 out_d[i+i] = 'A';
         out_d[i + i + 1] = 'B';
}
}
