#include <stdio.h>
#include <unistd.h>
#include <time.h>

#define SIZE 1000000

int main(){
    int arr[1024],sum=0;

    for (int i = 0; i < 1024; i++) {
        arr[i] = 42;
    }
    for (int i=0; i < 1024; i++){
        sum = sum+arr[i];
    };
    printf("%d",sum);
    return 0;
}