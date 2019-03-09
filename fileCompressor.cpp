#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char** argv) {
    if(argc != 2) {
	cout << "Invalid number of arguments. Usage is: ./histogram <filename>";
	return 0;
    }
    string fileName = argv[1];
    ifstream inFile(fileName);

    //Get the length of the file in characters
    int num_elements = 0;
    if(inFile) {
	inFile.seekg(0, inFile.end);
	num_elements = inFile.tellg();
	inFile.seekg(0, inFile.beg);
    }

    //Read file in character array
    unsigned char* h_input = new char[num_elements];
    inFile.read(h_input, num_elements);
    inFile.close();

    //Building frequency table
    int h_bins[37];
    
}
