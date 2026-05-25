#include <stdio.h>
#define PRINT_VECTOR_BUILD
#include "print_vector.h"

PRINT_VECTOR_API;

void print_vector_int(int * v, int len)
{
  for (int i = 0; i < len; i++) {
    printf("%d", v[i]);
    if (i == len - 1) {
      printf("\n");
    } else {
      printf(" ");
    }
  }
}

void print_vector_float(float * v, int len)
{
  for (int i = 0; i < len; i++) {
    printf("%f", v[i]);
    if (i == len - 1) {
      printf("\n");
    } else {
      printf(" ");
    }
  }
}
