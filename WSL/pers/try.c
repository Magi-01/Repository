#include <stdio.h>

// prints ASCII from 33 to 127 such that the user inputs a letter
// the programm then checks if the users input is equal to the letter 'Y'
// if it is it prints "you win" otherwise "you loose, retry"

int main() 
{
  int k,j,i=33;
  char f,t='_',l='|'; // declaring of char var fa

  printf("choose from one of the below\n\n");

  for(k=0;k<13;k=k+1)
    {
      for(j=0;j<11;j=j+1)
        {
          if((k==0)||(k==12))
            {
              printf(" %c%c%c",t,t,t);
              if(((k==0)||(k==12))&&((j==10)))
                {printf(" %c%c",t,t);}
            }

           if(((k>0)&&(k<12))&&((j==0)||(j==10)))
              {
                if((k==11)&&(j==10))
                {printf("                     ");}
                  printf("%c ",l);
              }

            if(((k>0)&&(k<12))&&((j>0)&&(j<10)))
              {
                if ((i>32)&&(i<128))
                printf(" %c;  ",i);
                i++;
              }
          //if ((k==9))
            //{printf()}
          if(j==10)
            {printf("\n");}
        }
    }
  
  

  printf("\n\nInsert your letter: "); // user input of first letter
  
  for(int i=0;i<10;i++)
    {
      scanf("%c" ,&f);
      //user input assigned to index pointer f
      // retry untill you get the right letter 'Y' i times
      if(f != 'Y')
      {
        printf("\n\nyou loose. you have %d tries left retry: ",10-i);
        scanf("%c" ,&f);
      }

      else
      {
        printf("\nyou win\n");
        break;
      }
      
      }
      

  return 0;
}