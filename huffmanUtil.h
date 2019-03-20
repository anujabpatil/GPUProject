#ifndef _huffmanutil_h
#define _huffmanutil_h

#include <iostream>
#include <string>
#include "huffmanNode.h"

#define NUM_BINS 40
#define SPACE_INDEX 36
#define PERIOD_INDEX 37
#define COMMA_INDEX 38
#define NEW_LINE 39

using namespace std;

/*
 *  * Checks whether the given file exists; if it does, prompts the user whether
 *   * they want to overwrite the file.  Returns true if the user does want to
 *    * overwrite, and false if not.
 *     */
bool confirmOverwrite(string filename);

/*
 *  * Returns the size of the given file in bytes.
 *   */
int fileSize(string filename);

/*
 *  * Reads the entire contents of the given input file and returns them as a string.
 *   */
string readEntireFileText(string filename);

HuffmanNode* buildEncodingTree(const unsigned int* freqArr);

string* buildEncodingArray(HuffmanNode* encodingTree);

void traverseTreeForBinaryCode(HuffmanNode* root, const string& code, string* codeArr);

int getMaxCodeLength(string* codeArr);

#endif

