
#ifndef _DECISION_TREE_H_
#define _DECISION_TREE_H_

#include <vector>
#include <deque>
// #include "data.h"

using namespace std;

struct Node {
    int ind;
    int fid;
    float split_val;
    int child[2];
    vector<int> id_list;
};

struct Datum{
    int id;
    vector<float> f;
    int label;
};

struct Fval{
    int id;
    float val;
    int label;
};

class DecisionTree{

public:
    int maxDepth;
    int n;
    int m;

    vector<Node> tree;
    deque<int> q;
    DecisionTree(vector<Datum> &data, vector<int> &labels, const int max_depth);
    ~DecisionTree();

    void print_tree(vector<int> &labels);

private:
    void expand(int index, vector<Datum> &data, vector<int> &labels);


    // float predict();

};

#endif
