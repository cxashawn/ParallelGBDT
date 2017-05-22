#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <float.h>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include "decision_tree.h"
using namespace std;

double total_time = 0.0;

DecisionTree::DecisionTree(vector<Datum> &data, vector<int> &labels, const int max_depth, const int n_threads){

    n = data.size();
    m = data.size() > 0 ? data[0].f.size() : 0;
    maxDepth = max_depth;
    nThreads = n_threads > 1 ? n_threads : 1;

    q.reserve(256);
    q.resize(0);

    omp_set_num_threads(nThreads);

    #pragma omp parallel
    {
        this->nThreads = omp_get_num_threads();
    }

    printf("Number of threads: %d\n", this->nThreads);

    Node root;
    root.ind = 0;
    root.split_val = -1;
    for(int i=0; i<data.size(); i++){
        root.id_list.push_back(i);
    }
    root.fid = -1;
    root.child[0] = -1;
    root.child[1] = -1;

    tree.push_back(root);
    q.push_back(root.ind);

    for(int i=1; i<maxDepth; i++){
        int size = q.size();
        split_q.resize(size);
        // TODO: Parallel here
        // #pragma omp parallel for schedule(dynamic, 1)
        double start_time = get_time();
        for (int j=0; j<size; j++){
            // int index = q.front();
            // q.pop_front();
            this->expand(j, data, labels);
        }
        total_time += get_time() - start_time;
        // TODO: adding updateTree func
        this->updateTree();
    }
}

DecisionTree::~DecisionTree(){};

bool compFval(Fval v1, Fval v2) {return (v1.val < v2.val);}

float info_gain(int left[2], int right[2], int sum[2]){

    float sum_ratio_0 = (double)sum[0] / (sum[0]+sum[1]);
    float sum_ratio_1 = (double)sum[1] / (sum[0]+sum[1]);
    float sum_score = -sum_ratio_0 * log2(sum_ratio_0) - sum_ratio_1 * log2(sum_ratio_1);

    float left_ratio_0 = (double)left[0] / (left[0]+left[1]);
    float left_ratio_1 = (double)left[1] / (left[0]+left[1]);
    float left_score = -left_ratio_0 * log2(left_ratio_0) - left_ratio_1 * log2(left_ratio_1);

    float right_ratio_0 = (double)right[0] / (right[0]+right[1]);
    float right_ratio_1 = (double)right[1] / (right[0]+right[1]);
    float right_score = -right_ratio_0 * log2(right_ratio_0) - right_ratio_1 * log2(right_ratio_1);

    float avg_ch_score = ((double)(left[0]+left[1])/(sum[0]+sum[1])) * left_score + ((double)(right[0]+right[1])/(sum[0]+sum[1])) * right_score;

    return (sum_score - avg_ch_score);
}

struct FeatureInfo{
    int b_split;
    int b_f;
    float b_ig;
};

bool compFinfo(FeatureInfo f1, FeatureInfo f2) {return (f1.b_ig > f2.b_ig);}


void DecisionTree::expand(int idx, vector<Datum> &data, vector<int> &labels){

    int index = q[idx];

    // printf("Expand Node %d\n", tree[index].ind);
    // int best_split = -1;
    // float best_ig = FLT_MIN;
    // int best_f = -1;
    vector<FeatureInfo> ft_info;
    ft_info.reserve(m);
    ft_info.resize(0);

    #pragma omp parallel for schedule(dynamic, 1)
    for(int k=0; k<m; k++){
        // printf("Check feature %d\n", k);
        float local_b_ig = FLT_MIN;
        int local_b_split = -1;

        vector<Fval> templist;
        for(int i=0; i<tree[index].id_list.size(); i++){
            Fval temp;
            temp.id = tree[index].id_list[i];
            temp.val = data[temp.id].f[k];
            temp.label = labels[temp.id];
            templist.push_back(temp);
        }
        sort(templist.begin(), templist.end(), compFval);
        int left[2];
        left[0] = 0; left[1] = 0;
        int right[2];
        right[0] = 0; right[1] = 0;
        int sum[2];
        sum[0] = 0; sum[1] = 1;
        for(int i=0; i<templist.size(); i++){
            right[templist[i].label]++;
            sum[templist[i].label]++;
        }
        for(int i=0; i<templist.size()-1; i++){
            left[templist[i].label]++;
            right[templist[i].label]--;
            float temp_ig = info_gain(left, right, sum);
            if(temp_ig > local_b_ig){
                local_b_ig = temp_ig;
                local_b_split = k;
                // best_f = k;
            }
        }
        ft_info[k].b_ig = local_b_ig;
        ft_info[k].b_split = local_b_split;
        ft_info[k].b_f = k;
        // printf("Current best info gain%.4f\n", best_ig);
    }

    sort(ft_info.begin(), ft_info.begin()+m, compFinfo);

    float best_ig = ft_info[0].b_ig;
    int best_f = ft_info[0].b_f;
    int best_split = ft_info[0].b_split;


    vector<Fval> templist;
    for(int i=0; i<tree[index].id_list.size(); i++){
        Fval temp;
        temp.id = tree[index].id_list[i];
        temp.val = data[temp.id].f[best_f];
        temp.label = labels[temp.id];
        templist.push_back(temp);
    }
    sort(templist.begin(), templist.end(), compFval);

    if(best_split == -1 || best_f == -1){
        printf("Error happens when splitting!\n");
        abort();
    }
    // Node left;
    // left.ind = tree.size();
    // left.split_val = -1;
    // left.fid = -1;
    // left.child[0] = -1; left.child[1] = -1;
    // for(int i=0; i<=best_split; i++){
    //     left.id_list.push_back(templist[i].id);
    // }
    // tree.push_back(left);

    //TODO: using splitInfo func
    split_q[idx].fid = best_f;
    split_q[idx].fval = templist[best_split].val;
    vector<int> left;
    vector<int> right;

    for (int i=0; i<=best_split; i++){
        left.push_back(templist[i].id);
    }
    split_q[idx].ch_id_lists.push_back(left);
    for (int i=best_split+1; i<templist.size(); i++){
        right.push_back(templist[i].id);
    }
    split_q[idx].ch_id_lists.push_back(right);
    // Node right;
    // right.ind = tree.size();
    // right.split_val = -1;
    // right.fid = -1;
    // right.child[0] = -1; right.child[1] = -1;
    // for(int i=best_split+1; i<templist.size(); i++){
    //     right.id_list.push_back(templist[i].id);
    // }
    // tree.push_back(right);

    // q.push_back(left.ind);
    // q.push_back(right.ind);

    // tree[index].child[0] = left.ind;
    // tree[index].child[1] = right.ind;
    // tree[index].fid = best_f;
    // tree[index].split_val = templist[best_split].val;

    // printf("Finish Expand Node %d\n", index);

}

void DecisionTree::updateTree(){

    int size = q.size();

    vector<int> new_q;

    for(int i=0; i<size; i++){
        int index = q[i];
        tree[index].fid = split_q[i].fid;
        tree[index].split_val = split_q[i].fval;

        Node left;
        left.ind = tree.size();
        left.fid = -1; left.split_val = -1;
        left.child[0] = -1; left.child[1] = -1;
        left.id_list = split_q[i].ch_id_lists[0];

        Node right;
        right.ind = tree.size() + 1;
        right.fid = -1; right.split_val = -1;
        right.child[0] = -1; right.child[1] = -1;
        right.id_list = split_q[i].ch_id_lists[1];

        tree[index].child[0] = left.ind;
        tree[index].child[1] = right.ind;

        tree.push_back(left);
        tree.push_back(right);
        new_q.push_back(left.ind);
        new_q.push_back(right.ind);
    }
    q = new_q;

}

void DecisionTree::print_tree(vector<int> &labels){
    deque<int> tempq;
    tempq.push_back(0);
    int lvl = 0;
    printf("Tree size %d\n", tree.size());
    while(!tempq.empty()){
        int size = tempq.size();
        printf("Level %d\n", lvl++);
        for(int i=0; i<size; i++){
            int ind = tempq.front();
            tempq.pop_front();
            Node temp = tree[ind];
            printf("Node %d id_list: ", temp.ind);
            for(int j=0; j<temp.id_list.size(); j++){
                printf("%d(%d), ", temp.id_list[j], labels[temp.id_list[j]]);
            }
            if(temp.child[0] != -1){
                tempq.push_back(temp.child[0]);
            }
            if(temp.child[1] != -1){
                tempq.push_back(temp.child[1]);
            }
            if(i < size-1){
                printf(" | ");
            }
        }
        printf("\n");
    }
}

float get_rand(){
    float r = ((double)rand() / RAND_MAX);
    return r;
}

void print_data(vector<vector<float> > &d, vector<int> &l){
    for(int i=0; i<d.size(); i++){
        // vector<float> this_v = d[i];
        for(int j=0; j<d[i].size(); j++){
            printf("f%d: %.4f ", j, d[i][j]);
        }
        printf("label: %d\n", l[i]);
    }
}

int main(int argc, char **argv){

    printf("------------------------------------\n");
    printf("Test input data\n");

    if (argc < 4) {
        printf("SHOULD BE: ./gbdt_ft #ofSamples #ofFeatures #ofThreads\n");
        abort();
    }

    int n_samples = atoi(argv[1]);
    int n_fts = atoi(argv[2]);
    int n_thds = atoi(argv[3]);



    vector< vector<float> > testdata;
    vector<int> labels;
    for (int i=0; i<100000; i++){
        vector<float> temp;
        for (int j=0; j<50; j++){
            float f = get_rand();
            temp.push_back(f);
        }
        testdata.push_back(temp);
        float prob = get_rand();
        if(prob > 0.5){
            labels.push_back(1);
        } else{
            labels.push_back(0);
        }
    }
    // print_data(testdata, labels);

    printf("------------------------------------\n");
    printf("Start building decision tree\n");
    vector<Datum> input;
    for(int i=0; i < testdata.size(); i++){
        Datum temp;
        temp.id = i;
        temp.f = testdata[i];
        temp.label = labels[i];
        input.push_back(temp);
    }
    DecisionTree *t = new DecisionTree(input, labels, 4, n_thds);

    printf("------------------------------------\n");
    printf("Finish building decision tree\n");
    printf("Total parallel time is %.4f\n", total_time);
    // t->print_tree(labels);

}
