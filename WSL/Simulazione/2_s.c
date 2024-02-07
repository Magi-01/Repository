#include <stdio.h>
//#include <math.h>



void ite_pari(double x)
{
  double a_n;
  double a_n_a = 2;
  double a_n_b = 1;
  double y = a_n_b;
  for (int i=0; i<=x; i++)
  {
    if (i==0)
      printf("sum 0: 1\n");
      
    else if (i==1)
      printf("sum 1: 2\n");
      
    else if ((i>=2)&&(i%2==0))
    {
      a_n_b = y;
      
      a_n = (i + 3 * (a_n_b - 2)) / (a_n_a);
      printf("The sum %d: %lf\n", i, a_n);

      a_n_b = a_n_a;
      a_n_a = a_n;
      y = a_n;
    }
    else if ((i>=2)&&(i%2!=0))
    {
      a_n = (i + 3 * (a_n_a - 2)) / (a_n_b);
      printf("The sum %d: %lf\n", i, a_n);

      a_n_a = a_n;
    }
  }
}

int main(void)
{
  double x;
  printf("Inserisci numero: ");
  scanf("%lf", &x);
  
  ite_pari(x);

  return 0;
}