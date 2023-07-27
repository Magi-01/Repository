#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

pthread_mutex_t locki = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
float i = 0;
int k = 0;
float *n;
int flag = 1;

void * f (void*argc)
{
    float z = 0;
    while(i < 1000000){
        pthread_mutex_lock(&locki);
        pthread_cond_wait(&locki,&cond);
    
        z = 8*i + (4*i + 1)/7;
        //printf("trying i = %d, n = 0\n",(int)i);
        if ((int)z == z){
            *(n+k) = z;
            //printf("n = %f\n",*(n+k));
            k++;
        }
        i = i + 1;

        pthread_mutex_unlock(&locki);
        pthread_cond_signal(&cond);
    }
    return n;
}

int main(int argc, char * argv[])
{
    n = (float *)malloc(1000000 * sizeof(float));
    pthread_t t1;
    pthread_t t2;
    pthread_t t3;
    pthread_t t4;
    pthread_t t5;
    pthread_t t6;
    pthread_create(&t1,NULL,f,NULL);
    pthread_create(&t2,NULL,f,NULL);
    pthread_create(&t3,NULL,f,NULL);
    pthread_create(&t4,NULL,f,NULL);
    pthread_create(&t5,NULL,f,NULL);
    pthread_create(&t6,NULL,f,NULL);
    pthread_join(t1,NULL);
    pthread_join(t2,NULL);
    pthread_join(t3,NULL);
    pthread_join(t4,NULL);
    
    for (int i = 0; i < k; i++){
        printf("i = %f\n",n[i]);
    }
    free(n);
    return 0;
}