#define BLOCK_SIZE 512
/*int max(int x, int y) 
{ 
  int retvalue = (x > y) ? x : y;
  return retvalue; 
}*/


__global__ void lz77kernel(char *in_d, char *out_d, int search_buffer_size, int uncoded_buffer_size) 
{
	int maximum = 0;
        int i = threadIdx.x + blockIdx.x * blockDim.x;

	int k;
	int j;
	int length;
	int offset;
	 //int maximum;

//	int total_size = search_buffer_size + uncoded_buffer_size;
//	int shared_data_size = (total_size-1) + BLOCK_SIZE;
	__shared__ char data[BLOCK_SIZE];

	data[threadIdx.x] = in_d[i];

if( threadIdx.x < 499)
{
	int search_buffer_first = threadIdx.x;
        int uncoded_first = threadIdx.x + search_buffer_size;

        for ( j = search_buffer_first; j < search_buffer_first + search_buffer_size; j++)
	{
		if( data[j] == data[uncoded_first])
		{
			for(k = (uncoded_first + 1); k <= uncoded_buffer_size; k++)
			{	
				int last_searchbuffer_match = j;		
				if(data[k] == data[j + 1])
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
			offset = ( k - j);	
			break;
			
		}
		else
		{
			length = 0;	
		}
	
	}

	if ( length == 0)
	{
		out_d[i + i] = 1;
		out_d[i + i + 1] = data[uncoded_first];
	}
	else
	{
	 	out_d[i + i] = offset;
                out_d[i + i + 1] = length;
	}
}
}
