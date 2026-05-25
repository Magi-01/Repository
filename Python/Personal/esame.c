#include <stdio.h>
#include <unistd.h>

int main(int argc, char * arvg[]){
    pid_t pid1 = fork();
    pid_t pid2 = fork();
    if (pid1 != 0)
    {
        pid1 = 1;
    }
    if (pid2 != 0)
    {
        pid2 = 1;
    }
    if (pid1 == 1){
        printf("a ");
    }else if (pid2 == 1) {
        printf("b ");
    }else{
        printf("c\n");
    }

    return 0;
}