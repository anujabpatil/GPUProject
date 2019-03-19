#include "huffmanUtil.h"
#include <stdio.h>

using namespace std;

bool confirmOverwrite(string filename) {
    if (!fileExists(filename)) {
        return true;
    } else {
	char option;
	cout << "File already exists. Overwrite ? (y/n): ";
	cin >> option;
	if(option == 'y'|| option == 'Y') return true;
	return false;
    }
}

int fileSize(string filename) {
    ifstream input;
    input.open(filename.c_str(), ifstream::binary);
    input.seekg(0, ifstream::end);
    return (int) input.tellg();
}

string readEntireFileText(string filename) {
    ifstream input;
    input.open(filename.c_str());
    return readEntireFileText(input);
}

