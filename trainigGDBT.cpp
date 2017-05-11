//binary classification, the y should either be -1 or 1;
#include <math.h>
#include <vector>
#include <cstdlib>
#include<queue>
#include<stack>
#include <iostream>
#include <ctime>
#include <unistd.h>
#include<assert.h>
#include <time.h>

using namespace std;
int numOfFeatures = 10;
int numofInstances= 500;
class Instance;
class DataInfo;
double gama = 1;
double lambda = 0.01;
int maxDepth;
int indexSort;


int convertToOneDimension(int rowNum, int colNum){
    return numOfFeatures * rowNum + colNum;
}

vector<Instance> random_sample;

template<class Type >
void swap(Type *  a, Type * b){
    Type  p = *a;
    *a = *b;
    *b = p;
}

class Instance{
public:
    Instance(){
        
    }
    Instance( vector <int > input, int output, int i ){
        x = input;
        y = output;
        instanceNumber = i;
    }
    
    vector <int > x;
    int y;
    int instanceNumber;
    
};

int generateInputNum(){
    //srand((unsigned)time());
    int res = rand();
    return res;
}
int generateOutput(){
    //srand((unsigned)time(1));
    double r = ((double) rand() / (RAND_MAX));
    if (r <= 0.5){
        return -1;
    }else{
        return 1;
    }

}
void printInput(){
    for (int i = 0; i < numofInstances; i++){
        vector <int > input = random_sample[i].x;
        int output =random_sample[i].y;
        cout<<random_sample[i].instanceNumber<<" : ";
        for (int j = 0; j < numOfFeatures; j++ ){
            cout<< input[j]<< ", ";
        }
        cout<< output<<endl;
    }
}

void generateRandomSamples(){
    for (int i = 1; i <= numofInstances; i++){
        vector <int > input;
        for (int j = 0 ; j < numOfFeatures; j++){
            input.push_back(generateInputNum());
            //usleep(1);
        }
        //int output = generateOutput();
        int output= -1;
        if (input[0]% 2==0){
            output = 1;
        }
        Instance ins = Instance(input, output, i);
        random_sample.push_back(ins);
    }
}
/* 1. score(this score is the predicted value if this node is the leaf)(weight)
 * 2. an array contains it current instance,  g(first order derivative ) and h (second order derivative) for each instance
 * 3. featureIndex that used to split from it parent node
 * 4. splitValue from parent 
 * 5.
 */
class SplitInfo {
    public:
    SplitInfo(){
        
    }
    /*
    SplitInfo(int weight, vector<Instance> curIns, vector<double> gi, vector<double>hi){
        curInstances = curIns;
        w = weight;
        g = gi;
        h = hi;
    }
     */
    
    SplitInfo(double weight, vector<DataInfo> data){
        w = weight;
        dataInfoVector = data;
        
    }
    SplitInfo(double weight, vector<DataInfo> data, int value, int index){
        w = weight;
        dataInfoVector = data;
        splitValue = value;
        featureIndex = index;
    }
  
    double w;
    int featureIndex;
    int splitValue;
    vector<DataInfo> dataInfoVector;
    
};

class DataInfo{
    public:
    DataInfo(Instance data){
        ins = data;
    }
    Instance ins;
    int g;
    int h;
};

/*for each node, it at least contains
 * 1. splitInfo: result information from last split
 * 2. a left node and a right node
 *
 */
class Node{
public:
    /*
    Node (int deep ){
        depth = deep;
    }
     */
    Node(SplitInfo information, int deep){
        info = information;
        left = NULL;
        right = NULL;
        depth = deep;
    }
    Node (){
        depth = -1;
        left = NULL;
        right = NULL;
    }
    
    //after find best split, update left and right child, feature index
    void updateNode(Node* l, Node* r ){
        left = l;
        right = r;
    }
    SplitInfo info;
    int depth;
    Node* left;
    Node* right;
};

Node* curNode;


double computeGain(double Gl, double Gr, double Hl, double Hr ){
    return 0.5 * (Gl*Gl/ (Hl + lambda) + Gr*Gr/ (Hr + lambda) - (Gl+Gr)*(Gl+Gr)/ (Hr + Hl+ lambda)) - gama;
}

//split instances at index i
double splitAndComputeGain(vector<DataInfo> data, int i ){
    double Gl = 0, Gr = 0 , Hl = 0 , Hr = 0 ;
    int lnum = i + 1;
    for (int m = 0 ; m < i + 1; m++){
        Gl += data[m].g;
        Hl += data[m].h;
    }
    
    for (int n = i + 1; n < data.size(); n++){
        Gr += data[n].g;
        Hr += data[n].h;
    }
    return computeGain(Gl, Gr, Hl, Hr );
    
    
    
}

double firstOrderDerivative(double y_p, int y){
    return 2* ( y_p -y );
}

// h is always equal to 2 for square loss
double secondOrderDerivative(){
    return 2;
}

//computer score for child node after find best split
double computeScore(vector<DataInfo> data){
    double G = 0;
    double H = 0;
    for (int i = 0; i<data.size();i++){
        G = G+ data[i].g;
        H = H + data[i].h;
    }
    //cout<< "g is: "<<G<< "H is "<< H<<endl;
    //cout<< "res is "<< -G/ (H + lambda)<<endl;
    return -G/ (H + lambda);
}

bool compareDataInfo(DataInfo point1, DataInfo point2){
    //cout<< "sorting "<<endl;
    return (point1.ins.x[indexSort] < point2.ins.x[indexSort]);
}

void sortData (vector<DataInfo>&data , int index){
    indexSort = index;
    //cout<< "index is "<<indexSort<<endl;
    sort(data.begin(),data.end(), compareDataInfo);
}

bool buildRootNode(Node& root);
bool buildOtherNode(Node* cur);

class DecisionTree{
    public:
    
    DecisionTree(){
        
    }
    //return true if successfully build a tree, false otherwise
    bool buildTree(){
        if (root.depth == -1){
            //cout<< "empty tree"<<endl;
            if (buildRootNode(root)){
                 curNode = &root;
                 //cout<< "built a root"<<endl;
                 currentRowNodes.push(curNode);
            }else {
                cout<<"no more tree to build"<<endl;
                return false;
            }
        }
        while(true){
            if (curNode->depth == maxDepth){
                cout<< "max depth reached"<<endl;
                break;
            }else {
                //cout<< "cur depth is "<< curNode.depth<<endl;
            }
            if (currentRowNodes.size()== 0){
                if (nextRowNodes.size()!=0){
                    swap(currentRowNodes,nextRowNodes);
                }else{
                    //cout<< "no current row nodes"<<endl;
                    break;
                }
            }
            
            curNode = currentRowNodes.top();
            currentRowNodes.pop();
            /*
            cout << "hi"<<endl;
            Node* m = curNode.left;
            Node* n = curNode.right;
            cout << "hello"<<endl;
            cout << "::current node left depth is sa aaaaa "<< curNode.left->depth<<endl;
            cout << "::current node right depth is sa aaaaa "<< curNode.right->depth<<endl;
            
            */
            if (buildOtherNode(curNode)){
                nextRowNodes.push(curNode->left);
                nextRowNodes.push(curNode->right);
            }

        }
        
        computeTrainingError();
       
        return true;
    }
    
    static pair<int, int> findBestSplit(Node* curNode){
        //cout<< "finding best split"<<endl;
        if (curNode->depth ==maxDepth){
            return make_pair(-1,0);
        }
        
        int featureIndex;
        int value;
        int maxGain = 0;
        for (int i = 0 ; i< numOfFeatures; i++){
            sortData(curNode->info.dataInfoVector ,i );
            for (int j = 0; j < curNode->info.dataInfoVector.size() - 1; j++){
                int gain = 0;
                //cout<< "gain is " <<gain<<endl;
                gain = splitAndComputeGain(curNode->info.dataInfoVector, j);
                //cout<< " after gain is " <<gain<<endl;
                if (maxGain < gain){
                    maxGain = gain;
                    featureIndex = i;
                    value = curNode->info.dataInfoVector[j].ins.x[i];
                }
                
            }
            
        }
        //cout << "feature index is " << featureIndex<< "value is " <<value<<endl
        if (maxGain == 0){
            return make_pair(-1,0);
        }else{
            cout << "max gain is " << maxGain<<endl;
        }
        return make_pair(featureIndex,value);
    }
    
//print tree structure : if it is a leaf leaf:( depth, weight); if it is a node: ( depth, weight, feature index, value)
    void printTree(){
        cout<< "Decision Tree:"<<endl;
        stack<Node> currentNodes;
        stack<Node> nextNodes;
        Node cur;
        if (root.depth == -1){
            //cout<< "empty tree"<<endl;
        }else{
            //cout<< "There is a tree"<<endl;
            currentNodes.push(root);
        }
        while(true){
            //cout<< "in whole loop "<<endl;
            if (currentNodes.size()== 0){
                if (nextNodes.size()!=0){
                    //cout<< "swapping"<<endl;
                    swap(currentNodes,nextNodes);
                }else{
                    //cout<< "finish"<<endl;
                    break;
                }
            }
            cur = currentNodes.top();
            currentNodes.pop();
            //printnode (depth, score)
            
            if (cur.left){
                nextNodes.push(*(cur.left));
                nextNodes.push(*(cur.right));
                cout<< "node: ( " << cur.depth<<", "<< cur.info.w<<", "<<cur.info.featureIndex<<", "<<cur.info.splitValue<<" )"<<endl;
            }else {
                cout<< "leaf : ( " << cur.depth<<", "<< cur.info.w<<" )"<<endl;
            }
        }
    }
    
    double predict(Instance ins){
        Node cur = root ;
        while(true){
            if (cur.left){
                int index = cur.info.featureIndex;
                int value = cur.info.splitValue;
                if (ins.x[index]<value){
                    cur = *(cur.left);
                }else {
                    cur = *(cur.right);
                }
                
            }else {
                return cur.info.w;
            }
        }
    }
    
    double computeTrainingError(){
        double res = 0;
        for (int i = 0; i < numofInstances; i++){
            double err = predict(random_sample[i])-random_sample[i].y;
            /*
             cout<< "predict is "<<predict(random_sample[i])<<endl;
             cout<< "real is "<< random_sample[i].y<<endl;
             cout<< "error is "<< err<<endl;
             */
            res+= err * err;
        }
        cout<< "training error after built this tree is "<< res/numofInstances<<endl;
        return res/ numofInstances;
    }
    
    
    Node root ;
    stack<Node*> currentRowNodes;
    stack<Node*> nextRowNodes;
    
    
};
vector<DecisionTree*> forest;

class GBDT{
    public:
    int numOfTrees;
    int minInstances;
    
    GBDT(int ntrees, int depth, int minIns){
        numOfTrees = ntrees;
        maxDepth = depth;
        minInstances = minIns;
    }
    void start(){
        for (int i = 0; i < numOfTrees; i++){
            DecisionTree* d = new DecisionTree();
            //cout<< "tyring to create a new decision tree"<<endl;
            if (!d->buildTree()){
                cout << "fail to built " << i+1<< "th tree"<<endl;
                break;
            }
            cout << "built " << i+1<< "th tree"<<endl;
            forest.push_back(d);
        }
    }
    static double predict(Instance ins){
        double res = 0;
        int n = forest.size();
        for (int i = 0; i < n; i++){
            res += forest[i]->predict(ins);
        }
        return res / n;
    }
    
    double computeTrainingError(){
        double res = 0;
        for (int i = 0; i < numofInstances; i++){
            double err = predict(random_sample[i])-random_sample[i].y;
            /*
            cout<< "predict is "<<predict(random_sample[i])<<endl;
            cout<< "real is "<< random_sample[i].y<<endl;
            cout<< "error is "<< err<<endl;
             */
            res+= err * err;
        }
        return res/ numofInstances;
    }
    
    double computeAccuracy(){
        double res = 0;
        for (int i = 0; i < numofInstances; i++){
            if (predict(random_sample[i])*random_sample[i].y>=0){
                res+=1;
            }
            /*
             cout<< "predict is "<<predict(random_sample[i])<<endl;
             cout<< "real is "<< random_sample[i].y<<endl;
             cout<< "error is "<< err<<endl;
             */
        }
        return res/ numofInstances;

    }
    
};

bool buildRootNode(Node& root){
    //pair<int, int> rootSplitRes;
    SplitInfo infor;
    vector<DataInfo> thisdataInfoVector;
    for(int i= 0; i < numofInstances; i++){
        Instance cur = random_sample[i];
        DataInfo d = DataInfo(cur);
        d.h = secondOrderDerivative();
        //fix here y_p
        if (forest.size()==0){
            d.g = firstOrderDerivative(0,cur.y );
        }else {
            d.g = firstOrderDerivative(GBDT::predict(cur),cur.y );
        }
        
        thisdataInfoVector.push_back(d);
    }

    infor.dataInfoVector = thisdataInfoVector;
    root.info = infor;
    /*
    double G = 0;
    double H = 0;
    for (int i = 0; i<infor.dataInfoVector.size();i++){
        G = G+ dataInfoVector[i].g;
        H = H + dataInfoVector[i].h;
    }
     */
    
    root.info.w = computeScore(thisdataInfoVector);

    
    //rootSplitRes = DecisionTree::findBestSplit(*root);
    //int featureNum = get<0>(rootSplitRes);
    //int splitValue = get<1>(rootSplitRes);
    //cout<< "feature num is "<< featureNum<< "value is " << splitValue<<endl;
    /*
    if (featureNum == -1){
        return false;
    }
    cout<< "found split for root"<<endl;
    cout<< "index featue is "<<featureNum<<endl;
    cout<< "value is "<< splitValue<<endl;
     */
    root.depth = 0;
    //root->info.featureIndex =  featureNum;
    //root->info.splitValue = splitValue;
    return true;
}

bool buildOtherNode(Node* cur){
    //cout<< "trying to build other node"<<endl;
    pair<int, int> splitres;
     //cout << "try to find best split for other node"<<endl;
    splitres = DecisionTree::findBestSplit(cur);
    //cout<< "getting split result"<<endl;
    if (get<0>(splitres) == -1){
        return false;
    }
   
    int featureNum = get<0>(splitres);
    int splitValue = get<1>(splitres);
    //update current node value
    //cout<< "index featue is "<< featureNum<<endl;
    //cout<< "value is "<<splitValue<<endl;
    cur->info.featureIndex = featureNum;
    cur->info.splitValue = splitValue;
    vector<DataInfo> ldata;
    vector<DataInfo> rdata;
    for(int i = 0 ; i < cur->info.dataInfoVector.size();i++){
        DataInfo d = cur->info.dataInfoVector[i];
        if (cur->info.dataInfoVector[i].ins.x[featureNum] < splitValue){
            ldata.push_back(cur->info.dataInfoVector[i]);
        }else{
            rdata.push_back(cur->info.dataInfoVector[i]);
        }
    }
    //cout<< "ldata size is "<<ldata.size()<<endl;
    double lscore = computeScore(ldata);
    double rscore = computeScore(rdata);
    //SplitInfo(int weight, vector<DataInfo> data, int value, int index){
    SplitInfo linfo = SplitInfo(lscore,ldata);
    SplitInfo rinfo= SplitInfo(rscore,rdata);
    Node* l = new Node(linfo, cur->depth+1);
    Node* r = new Node(rinfo, cur->depth+1);
    cur->left = l;
    cur->right = r;
    /*
    cout << "finish build other node, result:"<<endl;
    cout<< "other node left dpeth is "<<cur->left->depth<<"other node left  score is "<<cur->left->info.w<< endl;
    cout<< "other node right dpeth is "<<cur->right->depth<<"other node right  score is "<<cur->left->info.w<< endl;
     */

    return true;
}

int main(){
    generateRandomSamples();
    printInput();
    int treeNum = 10;
    GBDT mygdbt = GBDT(treeNum, 1, 10);
    mygdbt.start();
    cout<< "training error is: "<< mygdbt.computeTrainingError()<<endl;
    cout<< "accuracy is "<< mygdbt.computeAccuracy()<<endl;
    //clean up
    for (int i= 0; i < treeNum; i++){
        delete forest[0];
        forest.erase(forest.begin());
    }
    return 0;
}
