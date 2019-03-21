#include "huffmanUtil.h"
#include <stdio.h>
#include "huffmanNode.h"
#include <queue>
#include "huffmanNodeComparator.cu"
#include <fstream>
#include <cstring>

using namespace std;

/*bool confirmOverwrite(string filename) {
    if (!fileExists(filename)) {
        return true;
    } else {
	char option;
	cout << "File already exists. Overwrite ? (y/n): ";
	cin >> option;
	if(option == 'y'|| option == 'Y') return true;
	return false;
    }
}*/

int fileSize(string filename) {
    ifstream input;
    input.open(filename.c_str(), ifstream::binary);
    input.seekg(0, ifstream::end);
    return (int) input.tellg();
}

/*string readEntireFileText(string filename) {
    ifstream input;
    input.open(filename.c_str());
    return readEntireFileText(input);
}*/

HuffmanNode* buildEncodingTree(const unsigned int* freqArr) {
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

string* buildEncodingArray(HuffmanNode* encodingTree, int* codeLengths) {
    string* codeArr = new string[NUM_BINS];
    if(encodingTree != NULL) {
        string code = "";
        traverseTreeForBinaryCode(encodingTree, code, codeArr, codeLengths);
    }
    return codeArr;
}

void traverseTreeForBinaryCode(HuffmanNode* root, const string& code, string* codeArr, int* codeLengths) {
    if(root->isLeaf()) {
        if(root->character >= 97 && root->character <= 122) {
	    codeArr[root->character - 97] = code;
	    codeLengths[root->character - 97] = strlen(code.c_str());
	} else if(root->character >= 48 && root->character <= 57) {
	    codeArr[root->character - 22] = code;
	    codeLengths[root->character - 22] = strlen(code.c_str());
	} else if(root->character == 32) {
	    codeArr[SPACE_INDEX] = code;
	    codeLengths[SPACE_INDEX] = strlen(code.c_str());
	} else if(root->character == 46) {
	    codeArr[PERIOD_INDEX] = code;
	    codeLengths[PERIOD_INDEX] = strlen(code.c_str());
	} else if(root->character == 44) {
	    codeArr[COMMA_INDEX] = code;
	    codeLengths[COMMA_INDEX] = strlen(code.c_str());
	} else if(root->character == 10) {
 	    codeArr[NEW_LINE] = code;
	    codeLengths[NEW_LINE] = strlen(code.c_str());
	}
    } else {
        traverseTreeForBinaryCode(root->zero, code + '0', codeArr, codeLengths);
        traverseTreeForBinaryCode(root->one, code + '1', codeArr, codeLengths);
    }
}

int getMaxCodeLength(string* codeArr) {
    int maxLength = 0;
    for(int i = 0; i < NUM_BINS; i++) {
	int length = codeArr[i].size();
	if(length > maxLength) maxLength = length;
    }
    return maxLength;
}
