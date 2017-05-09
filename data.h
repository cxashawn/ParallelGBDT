#ifndef _DATA_H_
#define _DATA_H_

#include <vector>

struct Feature{
    int id;
    float value;
};

struct Datum{
    int ind;
    vector<feature> features;
    int label;
}

class Data{
public:
    int size;
    vector<datum> dataset;

    data(const int size, const vector<int[]> data);

    ~data();

};

#endif
