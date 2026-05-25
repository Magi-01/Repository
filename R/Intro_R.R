######## Esercizio #########
#1. Definisci il vettore y con gli elementi 8, 3, 5, 7, 6, 6, 8, 9, 2.
# Gli elementi di y sono minori di 5? 
y <- c(8,3,5,7,6,6,8,9,2)
y
all(y<5)
# Crea un nuovo vettore z con gli elementi di y minori di 5.
z <- c(y<5)
z <- which(z)
z <- y[z]
z


#2. Fornisci un esempio in cui valori logici sono convertiti in numerici 0-1
# utilizzando un operatore aritmetico
y <- c(1:4)
x <- c(seq(from=1, to=10, by=3))
z <- c(as.logical(x==y))
z <- z*1
z


#3. Crea un vettore logico di lunghezza 3. Quindi, moltiplica questo vettore
# tramite runif(3). Cosa succede?
v <- c(T,T,F)
v*runif(3)
# Sto moltiplicando ogni elemento del vettore v con 3 elementi random 
# di una distribuzione uniforme


#4.
x <- sample(10) < 4
x
which(x)
which(sample(10) < 4)
#   Cosa fa questo codice?
# Il primo passo assegna 10 elementi da 1 a 10 che pottrebbe non essere ordinati,
# poi assegna il valore booleano comparando i valori se < 4 per ogni elemento
# Il secondo passo stampa gli elementi di x, che sono booleani
# Il terzo passo cerca gli elementi booleani True di x e stampa la loro posizione
# Il quarto rifà la prima e terzo step in un unico funzione dando nuovi valori 
# posizionali per la bool True


#5. Definisci un vettore con valori 9, 2, 3, 9, 4, 10, 4, 11. Scrivi il
# codice per calcolare la somma dei tre valori più grandi (Suggerimento: usa
# la funzione rev).
v <- c(9,2,3,9,4,10,4,11)
v <- rev(sort(v))
v_add <- sum(v[1:3])
v_add


#6. Crea un vettore x di lunghezza 4. Cosa restituisce il codice?
#   x[c(TRUE, TRUE, NA, FALSE)]
x <- c(1,2,3,4)
x
x <- x[c(TRUE, TRUE, NA, T)]
x
# Restituisce i soli valori con True, e, se non è definito (nè True nè False),
# allora restituisce NA.
# mentre se False, viene troncata.

k <- c(1,2,3)
names(k) <- c("a", "b", "c")
k
