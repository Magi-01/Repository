#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

pthread_mutex_t lockx = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t locky = PTHREAD_MUTEX_INITIALIZER;

int x = 1000000;
int y = 1000000;

void * f (void*arg)
{
    pthread_mutex_lock(&lockx);
    x++;
    pthread_mutex_unlock(&lockx);
    pthread_mutex_lock(&locky);
    y += x;
    pthread_mutex_unlock(&locky);
    return NULL;
}


int main(int argc, char * argv[])
{
    pthread_t t1;
    pthread_t t2;
    pthread_t t3;
    pthread_t t4;
    pthread_create(&t1,NULL,f,NULL);
    pthread_create(&t2,NULL,f,NULL);
    pthread_create(&t3,NULL,f,NULL);
    pthread_create(&t4,NULL,f,NULL);
    pthread_join(t1,NULL);
    pthread_join(t2,NULL);
    pthread_join(t3,NULL);
    pthread_join(t4,NULL);
    printf("x = %d, y = %d\n",x,y);
    return 0;
}