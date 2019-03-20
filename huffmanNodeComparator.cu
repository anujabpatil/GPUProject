struct HuffmanNodeComparator 
{
    bool operator()(const HuffmanNode* lhs, const HuffmanNode* rhs) const
    {
        return lhs->count > rhs->count;
    }
};
