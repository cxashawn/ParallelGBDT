#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <float.h>
#include <algorithm>
#include <math.h>
#include "decision_tree.h"
using namespace std;

DecisionTree::DecisionTree(vector<Datum> &data, vector<int> &labels, const int max_depth){

    n = data.size();
    m = data.size() > 0 ? data[0].f.size() : 0;
    maxDepth = max_depth;

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
        for (int j=0; j<size; j++){
            int index = q.front();
            q.pop_front();
            this->expand(index, data, labels);
        }
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

void DecisionTree::expand(int index, vector<Datum> &data, vector<int> &labels){

    printf("Expand Node %d\n", tree[index].ind);
    int best_split = -1;
    float best_ig = FLT_MIN;
    int best_f = -1;
    for(int k=0; k<m; k++){
        // printf("Check feature %d\n", k);
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
            if(temp_ig > best_ig){
                best_ig = temp_ig;
                best_split = i;
                best_f = k;
            }
        }
        // printf("Current best info gain%.4f\n", best_ig);
    }

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
    Node left;
    left.ind = tree.size();
    left.split_val = -1;
    left.fid = -1;
    left.child[0] = -1; left.child[1] = -1;
    for(int i=0; i<=best_split; i++){
        left.id_list.push_back(templist[i].id);
    }
    tree.push_back(left);

    Node right;
    right.ind = tree.size();
    right.split_val = -1;
    right.fid = -1;
    right.child[0] = -1; right.child[1] = -1;
    for(int i=best_split+1; i<templist.size(); i++){
        right.id_list.push_back(templist[i].id);
    }
    tree.push_back(right);

    q.push_back(left.ind);
    q.push_back(right.ind);

    tree[index].child[0] = left.ind;
    tree[index].child[1] = right.ind;
    tree[index].fid = best_f;
    tree[index].split_val = templist[best_split].val;

    printf("Finish Expand Node %d\n", tree[index].ind);

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

int main(int arvc, char **argv){


    printf("------------------------------------\n");
    printf("Test input data\n");

    vector< vector<float> > testdata;
    vector<int> labels;
    for (int i=0; i<50; i++){
        vector<float> temp;
        for (int j=0; j<5; j++){
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
    DecisionTree *t = new DecisionTree(input, labels, 3);

    printf("------------------------------------\n");
    printf("Finish building decision tree\n");

    t->print_tree(labels);

}
