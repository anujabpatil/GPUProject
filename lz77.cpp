#include <stdio.h>
#include <stdint.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char** argv) 
{

    int s_b_s;
    int u_b_s;
    int j;
    j = 0;
    int k;
    k = 0;
    s_b_s = 8;
    u_b_s = 6;
    int length;
    length = 0;
    int offset;
    offset = 0;

    if(argc != 2) 
    {
        cout << "Invalid number of arguments. Usage is: ./compressFile <filename>\n";
        return 0;
    }
    string fileName = argv[1];
    ifstream inFile(fileName.c_str());

    unsigned int num_elements = 0;
    if(inFile) {
        inFile.seekg(0, inFile.end);
        num_elements = inFile.tellg();
        inFile.seekg(0, inFile.beg);
    }

    
    char* h_input = new char[num_elements];
    inFile.read(h_input, num_elements);
    inFile.close();
    char* h_output = new char[512];

while (j < num_elements)
{
    for(int y = j; y < (j + s_b_s); y++)
    {
        if (h_input[y] == h_input[y + s_b_s])
            {
                int last_buffer_match = j;
                for (int i = ( j + s_b_s ); i < (j + (s_b_s + u_b_s)); i++)
                {
                    if (h_input[i] == h_input[last_buffer_match])
                     {
                        length++;
                        last_buffer_match++;
                     }
                     else
                     {
                        last_buffer_match = 0;
                        break; 
                     }
                }
               if (length >= 3)
               {
                     //   offset = ( - j);
                        j = (length + j);
                        break;
               }
                else
               {
                        length = 0;
               }
            }  
    }   
        if (length == 0)
        {
                int x;
                h_output[k] = x + '0';
                h_output[k + 1] = 'A';
                j = j + 1;
        }
        else
        {
                char a = offset + '0' ;
                char b = length + '0';
                h_output[k] = a;
                h_output[k + 1] = b;
        }
        k = k + 2;
}
for(int l= 0; l<k; l++)
{
    printf("%c \n", h_output);
}
printf("%d\n",k);
}











