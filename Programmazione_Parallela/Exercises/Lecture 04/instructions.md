# Esercitazione 01

_7 Ottobre 2024_

Lo scopo di questa esercitazione è imparare a spezzare un progetto in più file C coi relativi file header e scrivere un makefile adeguato per gestire il progetto risultante. Per fare questo dovete compiere le seguenti operazioni:
- Spezzare il file ```all_together.c``` in un file ```main.c``` e separare in due file aggiuntivi l'implementazione di albero binario e quella di nodo dell'albero, rispettivamente. Sarà necessario creare i rispettivi file header.
- Creare un Makefile per la compilazione del progetto risultante. Il Makefile dovrà rispettare i seguenti requisiti:
  - Permettere di impostare facilmente le opzioni di compilazione.
  - Permettere di rimuovere tutti i file intermedi e gli eseguibili risultanti tramite il target ```clean```.
  - Creare una unica libreria (```libbst```) contenente sia l'implementazione di un albero binario che quella dei nodi dell'albero.
  - Permettere la scelta tramite una variabile ```SHARED``` se creare la libreria come statica o dinamica e modificare la generazione del target ```main``` di conseguenza.

### Extra

In aggiunta a questo, è possibile espandere il progetto nel seguente modo:
- Creare un file aggiuntivo ```print_tree.c``` (e relativo header) per stampare un albero binario in modo che dato un nodo $x$:
  - Se $x$ è ```NULL``` stampare solo un punto (i.e., ```printf(".");```).
  - Se $x$ è un nodo foglia stampare semplicemente il valore della chiave.
  - Altrimenti stampare come segue: ```([valore nella chiave] [sottoalbero sinistro] [sottoalbero destro])```.
- Aggiungere il file corrispondente alla libreria (statica o dinamica) generata tramite makefile e modificare il file ```main.c``` per far stampare l'albero generato dalle funzioni ```s_test``` e ```r_test```.