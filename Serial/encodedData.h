#ifndef _encodeddata_h
#define _encodeddata_h

#include <fstream>

class EncodedData {

    public:
	EncodedData(ofstream &outfile, int appendedZeroBits);
	ofstream& getEncodedFile();
	int getAppendedZeroBits();

    private:
	ofstream outfile;
	int appendedZeroBits;
};

#endif

