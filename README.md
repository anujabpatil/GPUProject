# GPUProject
Project description: 
 
We are implementing the DEFLATE algorithm which is used for G-Zip compression.            Compression saves bandwidth and speeds up the task. DEFLATE constitutes LZ77           algorithm and Huffman coding. 
 
Data Pre-processing: We will be reading text files as input. In C, we have FILE data structure to read the input                   file. We will read the file character-by-character and store them in character array. We              will copy this array to the device memory and then call the two kernels. 
 
Overview: We will first divide the input file into blocks which will be assigned to each thread block.                 We will apply LZ77 algorithm to each block. We will use two kernels - one for LZ77 and                  the other for Huffman coding. After LZ77 compression is done, we give the output from               that algorithm to Huffman coding algorithm. In Huffman kernel, we will divide this input              into blocks and assign each block to GPU block. Then, we will apply Huffman encoding               algorithm parallely to the blocks. 
 
Lempel-Ziv 77 (LZ77): LZ77 algorithm is dictionary based and uses it to store scanned substrings. It uses the               concept of a window buffer, also called as slider window, to serially move over the data                stream. The window buffer is divided into 2 parts - the search buffer and look-ahead               buffer. 
 
LZ77 algorithm starts by scanning the string or characters from beginning. As it scans,              each character will be slided into the look-ahead buffer. Then, that character is             searched in the search buffer. If the search is successful, the character in the              look-ahead buffer is replaced by <o,l,d> where o = offset, l = length and d = deviating                 character. The offset is the number of jumps that are required to go back to that                character in the search buffer. Length is the maximum number of characters that are              matched from the look-ahead buffer and search buffer, after the offset. ‘d’ is the              character which is different from the previous sequence and from which the next search              will begin. When the same word appears in the text again, it is not stored in the history                  
one more time but extracted with the help of the buffer. This reduces storage space and                makes the task faster. The output of LZ77 algorithm is called back-reference tokens. 
 
Parallel implementation of LZ77: In order to perform parallel computation, the LZ77 is divided into two levels. In the first                level, the input data stream is divided into equally sized chunks that are assigned to               each thread block to be compressed independently. In the second level, each thread is              assigned a different starting point in the chunk to perform a parallelized operation. 
 
Huffman encoding: In Huffman coding, the resulting back-reference tokens are converted to sequences of            codewords. In this algorithm, we generate Huffman trees based on the frequency of             each character and prefix sums calculations. Then we traverse the tree to determine             codewords for each character. 
 
Parallel implementation of Huffman encoding: For parallel computation of the Huffman encoding, first we generate the Huffman tree             serially on the host. Then, the prefix sum computation, encoded byte and compressed             bit stream generation are all done on the GPU parallely.
