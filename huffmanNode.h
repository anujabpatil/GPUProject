#ifndef _huffmannode_h
#define _huffmannode_h

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#define NOT_A_CHAR 257

using namespace std;

class HuffmanNode {

public:
    int character;       // character being represented by this node
    int count;           // number of occurrences of that character
    HuffmanNode* zero;   // 0 (left) subtree (NULL if empty)
    HuffmanNode* one;    // 1 (right) subtree (NULL if empty)

/*
 * Constructs a new node to store the given character and its count,
 * along with the given child pointers.
 */
    HuffmanNode(int character = NOT_A_CHAR, int count = 0,
                HuffmanNode* zero = NULL, HuffmanNode* one = NULL);

//Copy constructor
HuffmanNode(const HuffmanNode& other);

//Assignment operator overloading
HuffmanNode& operator=(HuffmanNode& other);

//Destructor
~HuffmanNode();

/*
 * Returns true if this node is a leaf (has NULL children).
 */
    bool isLeaf() const;

/*
 * Returns a string representation of this node for debugging.
 */
    string toString() const;
};

/*
 * Prints an indented horizontal view of the tree of HuffmanNodes with the given
 * node as its root.
 * Can optionally show the memory addresses of each node for debugging.
 */
static void printSideways(HuffmanNode* node, bool showAddresses = false, string indent = "");

/*
 * Stream insertion operator so that a HuffmanNode can be printed for debugging.
 */
ostream& operator <<(ostream& out, const HuffmanNode& node);

#endif
