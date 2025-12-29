#include <stdio.h>
//#include <math.h>



int ite_overlap_size(int i,int j, int t,int u)
{
  int cont;
  int m = 0;
  if ((i<t)&&(j<u))
  {
    for (int x=t; x<=j; x++)
    {
      cont = is_inside (x,i,u);
      if (cont == 0) m++;
    }
  }
  if ((i>t)&&(j>u))
  {
    for (int x=i; x<=t; x++)
    {
      cont = is_inside (x,t,j);
      if (cont == 0) m++;
    }
  }
  if ((i<t)&&(j>u))
  {
    for (int x=i; x<=j; x++)
    {
      cont = is_inside (x,i,j);
      if (cont == 0) m++;
    }
  }
  if ((i>t)&&(j<u))
  {
    for (int x=t; x<=i; x++)
    {
      cont = is_inside (x,t,i);
      if (cont == 0) m++;
    }
  }
  return m;
}

int is_inside (int x,int y,int z)
{
  if ((x>=y)||(x<=z))
    return 0;
}

int main()
{
  int i,j,u,t;
  printf("Inserisci primo elemento del primo intevallo: ");
  scanf("%d", &i);
  printf("Inserisci secondo elemento del primo intervallo: ");
  scanf("%d", &j);
  printf("Inserisci primo elemento del secondo intervallo: ");
  scanf("%d", &t);
  printf("Inserisci secondo elemento del secondo intervallo: ");
  scanf("%d", &u);
  
  int m = ite_overlap_size(i,j,t,u);

  printf("I numeri di elementi intersecatti sono: %d\n",m);

  return 0;
}