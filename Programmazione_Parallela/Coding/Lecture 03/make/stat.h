#ifndef _STAT_H
#define _STAT_H

#if defined(_STAT_BUILD)
#  if defined(_WIN32)
#    define _STAT_API __declspec(dllexport)
#  elif defined(__ELF__)
#    define _STAT_API __attribute__ ((visibility ("default")))
#  else
#    define _STAT_API
#  endif
#else
#  if defined(_WIN32)
#    define _STAT __declspec(dllimport)
#  else
#    define _STAT_API
#  endif
#endif

_STAT_API;

struct stat {
  int min;
  int max;
  float avg;
};

struct stat compute_stats(int * v, int len);

#endif
