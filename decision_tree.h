
#ifndef _DECISION_TREE_H_
#define _DECISION_TREE_H_

#include <vector>
#include <deque>
#include <sys/time.h>
#include <ctime>
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

struct splitInfo{
    // int child[2];
    int fid;
    float fval;
    vector<vector<int> > ch_id_lists;
};

class DecisionTree{

public:
    int maxDepth;
    int n;
    int m;
    int nThreads;

    vector<Node> tree;
    vector<splitInfo> split_q;
    vector<int> q;

    DecisionTree(vector<Datum> &data, vector<int> &labels, const int max_depth, const int n_threads);
    ~DecisionTree();

    void print_tree(vector<int> &labels);

private:
    void expand(int index, vector<Datum> &data, vector<int> &labels);
    void updateTree();


    // float predict();

};

double get_time(){
    struct timeval   tp;
    struct timezone  tzp;
    gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

#endif
