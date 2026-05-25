#include <stdio.h>
//#include <math.h>



int ric_som(int x)
{
  if (x == 1)
    return 1;
  else
    return ric_som(x);
}

int main()
{
  int x;
  printf("Inserisci numero: ");
  scanf("%d", &x);
  
  int m = ric_som(x);
  printf ("The sum is: %d\n", m);

  
  return 0;
}