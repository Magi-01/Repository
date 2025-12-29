#include <stdio.h>
//#include <math.h>

int ric_pot(int x)
{
  int pot = 2;
  if(x == 1)
    return 2;
  else
    return pot * ric_pot(x-1);
}

int ric_som(int x)
{
    if (x == 1)
       return 1;
    return x + ric_som(x - 1);

}
int op_ric(int x)
{
    if (x == 0)
        return 1;
    return ric_pot(x) * ric_som(x);
}

int main()
{
    int x;
    printf("Inserisci numero: ");
    scanf("%d", &x);

    int m = op_ric(x);
    printf("The sum is: %d\n", m);

    return 0;
}