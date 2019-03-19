#include "encodedData.h"
#include <stdio.h>

using namespace std;

EncodedData::EncodedData(ofstream &outfile, int appendedZeroBits) {
    this->outfile = outfile;
    this->appendedZeroBits = appendedZeroBits;
}

ofstream& EncodedData::getEncodedFile() {
    return outfile;
}

int EncodedData::getAppendedZeroBits() {
    return appendedZeroBits;
}
