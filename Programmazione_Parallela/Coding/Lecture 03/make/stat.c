#include <stdio.h>
#define _STAT_BUILD
#include "stat.h"

_STAT_API;

struct stat compute_stats(int *v, int len)
{
  struct stat s;
  s.min = v[0];
  s.max = v[0];
  s.avg = 0;
  for (int i = 0; i < len; i++) {
    if (v[i] < s.min) {
      s.min = v[i];
    }
    if (v[i] > s.max) {
      s.max = v[i];
    }
    s.avg += v[i];
  }
  s.avg /= len;
  return s;
}
