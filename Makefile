
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = compressFile
OBJ	        = fileCompressor.o support.o huffmanNode.o huffmanUtil.o

default: $(EXE)

fileCompressor.o: fileCompressor.cu kernel.cu support.h huffmanNode.h huffmanUtil.h
	$(NVCC) -c -o $@ fileCompressor.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

huffmanNode.o: huffmanNode.cu huffmanNode.h
	$(NVCC) -c -o $@ huffmanNode.cu $(NVCC_FLAGS)

huffmanUtil.o: huffmanUtil.cu huffmanUtil.h huffmanNodeComparator.cu
	$(NVCC) -c -o $@ huffmanUtil.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
