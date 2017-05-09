#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv){

    printf("Test random\n");

    float ch1 = (double)13 / 17;
    float ch2 = (double)4 / 17;


    float ret = -ch1 * log2(ch1) - ch2 * log2(ch2);


    printf("Result is %.4f\n", ret);

    return 1;



    int cnt = 20;
    while(cnt-- > 0){
        double r = ((double)rand() / RAND_MAX);
        if (r > 0.5){
            printf("In class 1\n");
        } else{
            printf("In class 2\n");
        }
    }
}
