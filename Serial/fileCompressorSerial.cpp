#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "huffmanNode.h"
#include "huffmanUtil.h"
#include "encodedData.h"
#include <bitset>
#include <queue>
#include "huffmanNodeComparator.cpp"

#define NUM_BINS 40
#define SPACE_INDEX 36
#define PERIOD_INDEX 37
#define COMMA_INDEX 38
#define NEW_LINE 39

using namespace std;

void calculateFrequencyArr(char* input, int num_elements, unsigned int* h_bins);
HuffmanNode* buildEncodingTree(const unsigned int* freqArr);
string* buildEncodingArray(HuffmanNode* encodingTree);
void traverseTreeForBinaryCode(HuffmanNode* root, const string& code, string* codeArr);
EncodedData encodeData(char* input, int num_elements, const string* encodingArray);

int main(int argc, char* argv[]) {

    //Timer timer;
    if(argc != 2) {
	cout << "Invalid number of arguments. Usage is: ./compressFileSerial <filename>\n";
	return 0;
    }

    string fileName = argv[1];
    ifstream inFile(fileName.c_str());

    //Get the length of the file in characters
    /*unsigned int num_elements = 0;
    if(inFile) {
	inFile.seekg(0, inFile.end);
	num_elements = inFile.tellg();
	inFile.seekg(0, inFile.beg);
    }*/
    int num_elements = fileSize(fileName);

    char* h_input = new char[num_elements];
    inFile.read(h_input, num_elements);
    inFile.close();

    unsigned int* h_bins = new unsigned int[NUM_BINS];

    //Calculate frequency array
    calculateFrequencyArr(h_input, num_elements, h_bins);

    //Build encoding tree
    HuffmanNode* encodingTree = buildEncodingTree(h_bins);

    //Build encoding array
    string* codeArr = buildEncodingArray(encodingTree);

    //Encode data
    EncodedData encodedData = encodeData(h_input, num_elements, codeArr);
    cout << "Actual file size: " << num_elements << endl;
    cout << "Encoded file size: " << fileSize("outfile.bin") << endl;
}

void calculateFrequencyArr(char* input, int num_elements, unsigned int* h_bins) {
    for(int i = 0; i < num_elements; i++) {
        if(input[i] >= 97 && input[i] <= 122) {
	    h_bins[input[i]-97] = h_bins[input[i]-97] + 1;
        } else if(input[i] >= 48 && input[i] <= 57) {
       	    h_bins[input[i]-22] = h_bins[input[i]-22] + 1;
        } else if(input[i] == 32) {
    	    h_bins[SPACE_INDEX] = h_bins[SPACE_INDEX] + 1;
        } else if(input[i] == 46) {
	    h_bins[PERIOD_INDEX] = h_bins[PERIOD_INDEX] + 1;
        } else if(input[i] == 44) {
	    h_bins[COMMA_INDEX] = h_bins[COMMA_INDEX] + 1;
        } else if(input[i] == 10) {
	    h_bins[NEW_LINE] = h_bins[NEW_LINE] + 1;
        }
    }
}


HuffmanNode* buildEncodingTree(const unsigned int* freqArr) {
    //PriorityQueue<HuffmanNode*> pqueue;
    priority_queue<HuffmanNode*, vector<HuffmanNode*>, HuffmanNodeComparator> pqueue;
    for(int i = 0; i < NUM_BINS; i++) {
        if(freqArr[i] == 0) continue;
        HuffmanNode* node = new HuffmanNode;
        if(i >= 0 && i <= 25) {
            node->character = i + 97;
        } else if(i >= 26 && i <= 35) {
            node->character = i + 22;
        } else if(i == 36) {
            node->character = 32;
        } else if(i == 37) {
            node->character = 46;
        } else if(i == 38) {
            node->character = 44;
        } else if(i == 39) {
            node->character = 10;
        }
        node->count = freqArr[i];
        node->zero = node->one = NULL;
        pqueue.push(node);
    }
    while(pqueue.size() != 1) {
    	HuffmanNode* leftSubTree = pqueue.top();
	pqueue.pop();
       	HuffmanNode* rightSubTree = pqueue.top();
	pqueue.pop();
       	HuffmanNode* sumNode = new HuffmanNode;
    	sumNode->character = NOT_A_CHAR;
        sumNode->count = leftSubTree->count + rightSubTree->count;
	sumNode->zero = leftSubTree;
  	sumNode->one = rightSubTree;
        pqueue.push(sumNode);
    }
    HuffmanNode* minNode = pqueue.top();
    pqueue.pop();
    return minNode;
}

string* buildEncodingArray(HuffmanNode* encodingTree) {
    string* codeArr = new string[NUM_BINS];
    if(encodingTree != NULL) {
        string code = "";
        traverseTreeForBinaryCode(encodingTree, code, codeArr);
    }
    return codeArr;
}

void traverseTreeForBinaryCode(HuffmanNode* root, const string& code, string* codeArr) {
    if(root->isLeaf()) {
        if(root->character >= 97 && root->character <= 122) codeArr[root->character - 97] = code;
        else if(root->character >= 48 && root->character <= 57) codeArr[root->character - 22] = code;
        else if(root->character == 32) codeArr[SPACE_INDEX] = code;
        else if(root->character == 46) codeArr[PERIOD_INDEX] = code;
        else if(root->character == 44) codeArr[COMMA_INDEX] = code;
        else if(root->character == 10) codeArr[NEW_LINE] = code;
    } else {
        traverseTreeForBinaryCode(root->zero, code + '0', codeArr);
        traverseTreeForBinaryCode(root->one, code + '1', codeArr);
    }
}

/*void encodeData(istream& input, const string* encodingArray, obitstream& output) {
    int ch;
    while(true) {
        ch = input.get();
        if(ch == EOF) break;
        string code;
        if(ch >= 97 && ch <= 122) code = encodingArray[ch - 97];
        else if(ch >= 48 && ch <= 57) code = encodingArray[root->character - 22];
        else if(ch == 32) code = encodingArray[SPACE_INDEX];
        else if(ch == 46) code = encodingArray[PERIOD_INDEX];
        else if(ch == 44) code = encodingArray[COMMA_INDEX];
        else if(ch == 10) code = encodingArray[NEW_LINE];
        for(int i = 0; i < code.length(); i++) {
            if(code[i] == '0') output.writeBit(0);
            else output.writeBit(1);
        }
    }
}*/

EncodedData encodeData(char* input, int num_elements, const string* encodingArray) {
    int ch;
    string fileCodeStr = "";
    for(int i = 0; i < num_elements; i++) {
	ch = input[i];
	if(ch == EOF) break;
	if(ch >= 97 && ch <= 122) fileCodeStr = fileCodeStr + encodingArray[ch - 97];
        else if(ch >= 48 && ch <= 57) fileCodeStr = fileCodeStr + encodingArray[ch - 22];
        else if(ch == 32) fileCodeStr = fileCodeStr + encodingArray[SPACE_INDEX];
        else if(ch == 46) fileCodeStr = fileCodeStr + encodingArray[PERIOD_INDEX];
        else if(ch == 44) fileCodeStr = fileCodeStr + encodingArray[COMMA_INDEX];
        else if(ch == 10) fileCodeStr = fileCodeStr + encodingArray[NEW_LINE];
    }
    ofstream outfile("outfile.bin", ios::binary);
    int appendedZeroBits = 0;
    for(int i = 0; i < fileCodeStr.length(); i++) {
	int count = 0;
	string bitStr = "";
	while(count < 8 && i < fileCodeStr.length()) {
	    bitStr = bitStr + fileCodeStr[i];
	    count++;
	    i++;
	}
	if(i == fileCodeStr.length() && count < 8) {
	    while(count < 8) {
		bitStr = "0" + bitStr;
		appendedZeroBits++;
		count++;
	    }
	}
	//convert string to binary
	bitset<8> binInt(bitStr);
	unsigned long n = binInt.to_ulong();
	outfile.write(reinterpret_cast<const char*>(&n), sizeof(n));
    }
    EncodedData encodedData(outfile, appendedZeroBits);
    return encodedData;
}
