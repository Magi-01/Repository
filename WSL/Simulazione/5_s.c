#include <stdio.h>
//#include <math.h>

int primo(int n, int i)
{
    if (n == 1) return 0;
    if ((i % n) == 0)
        return 1;
    else if ((i % n) != 0)
        return primo(n - 1, i);
}

int conta(int n, int i)
{
    int con = 0;
    int check = primo(i - 1, i);
    if (i == n) return 1;
    else if ((i != n) && (check == 0))
    {
        return 1 + conta(n, i - 1);
    }

}




int main()
{
    int n, i;
    printf("Inserisci minimo: ");
    scanf("%d", &n);
    printf("Inserisci massimo: ");
    scanf("%d", &i);

    int k = conta(n, i);
    if (k == 0)
        printf("non ci sono primi.\n");
    else
        printf("I numeri di primi sono: %d\n", k);

    return 0;
}