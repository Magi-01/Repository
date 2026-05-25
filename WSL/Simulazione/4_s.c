#include <stdio.h>
//#include <math.h>



void ite_pari(double x)
{
  int a_n;
  int a_n_a = 1;
  int a_n_b = 2;
  int y = a_n_b;
  for (int i=1; i<=x; i++)
  {
    if (i==1)
      printf("sum 1: 2\n");
      
    else if (i==2)
      printf("sum 2: 1\n");
      
    else if ((i>=3)&&(a_n_a%2==0))
    {
      a_n = (i - a_n_a) * (a_n_b);
      printf("The sum %d: %d\n", i, a_n);

      a_n_a = a_n;
      
    }
    else if ((i>=3)&&(a_n_a%2!=0))
    {
      a_n_b = y;

      a_n = (i - a_n_b) * (a_n_a);
      printf("The sum %d: %d\n", i, a_n);
      
      a_n_b = a_n_a;
      a_n_a = a_n;
      y = a_n;
      
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