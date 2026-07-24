#ifndef _PRINT_VECTOR_H
#define _PRINT_VECTOR_H

#if defined(PRINT_VECTOR_BUILD)
#  if defined(_WIN32)
#    define PRINT_VECTOR_API __declspec(dllexport)
#  elif defined(__ELF__)
#    define PRINT_VECTOR_API __attribute__ ((visibility ("default")))
#  else
#    define PRINT_VECTOR_API
#  endif
#else
#  if defined(_WIN32)
#    define PRINT_VECTOR __declspec(dllimport)
#  else
#    define PRINT_VECTOR_API
#  endif
#endif

PRINT_VECTOR_API;

void print_vector_int(int * v, int len);

void print_vector_float(float * v, int len);

#endif
