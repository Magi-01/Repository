#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

pthread_mutex_t l1,l2 = PTHREAD_MUTEX_INITIALIZER;

void * thread1(void * arg)
{
  pthread_mutex_lock(&l1);
  printf("acquisito l1 in t1\n");
  pthread_mutex_lock(&l2);
  printf("acquisito l2 in t1\n");
  for (int i = 0; i < 33; i++)
  {
    
  }
  

  return NULL;
}

void * thread2(void * arg)
{
  pthread_mutex_lock(&l2);
  printf("acquisito l2 in t2\n");
  pthread_mutex_unlock(&l2);
  printf("rilasciato l2 in t2\n");
  return NULL;
}

void * thread3(void * arg)
{
  pthread_mutex_lock(&l1);
  printf("acquisito l1 in t3\n");
  pthread_mutex_unlock(&l1);
  printf("rilasciato l1 in t3\n");
  return NULL;
}

int main(int argc, char * argv[])
{
  int * n = (void*)malloc(sizeof(int) * 100);
  pthread_t threads[3];
  pthread_create(&threads[0], NULL, thread1, n);
  pthread_create(&threads[1], NULL, thread2, n);
  pthread_create(&threads[2], NULL, thread3, n);
  pthread_join(threads[0], NULL);
  pthread_join(threads[1], NULL);  
  pthread_join(threads[2], NULL); 
}
