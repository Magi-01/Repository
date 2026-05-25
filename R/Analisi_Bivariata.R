#' Due variabili: uno da analizzare e uno di controllo
#' 
#' Es: categoriale (il tipo di scuola) e quantitativa (il voto)
#' -> target (dato il tipo di scuola, prevedo quanto con maggior
#' probabilità prenderai)
#' 
#' Es: quantitativo (il voto) e quantitativo (lo stipendio) 
#' -> target (quanto prendo con voto alto)
#' 
#' => Una Matrice 2*n
#'
par(mfrow=c(2,2))
set.seed(1111)
dat1<-rnorm(50)
plot(ecdf(dat1),cex=.5,
     main="confronto ripartizione empirica e teorica gaussiana")
curve(pnorm(x), add=TRUE)


dat2<-rt(50,2)
plot(ecdf(dat2), cex=.5, verticals=TRUE,
     main="confronto ripartizione empirica dati tratti da t con teorica gaussiana")
curve(pnorm(x), add=TRUE)


par(mfrow=c(1,2))
dat3<-rnorm(100) # generiamo dati da una gaussiana standard
hist(dat3,prob=T, xlim=c(-5,5), ylim=c(0,0.6))
# sovrapponiamo la funzione di densità della gaussiana standard
curve(dnorm(x,0,1), col=2, lwd=2, add=TRUE)
# Ora aumentiamo la numerosità dei dati
dat32<-rnorm(1000) # generiamo dati da una gaussiana standard
hist(dat32,prob=T, xlim=c(-5,5), ylim=c(0,0.6))
# sovrapponiamo la funzione di densità della gaussiana standard
curve(dnorm(x,0,1), col=2, lwd=2, add=TRUE)


dat4<-rt(1000,1) # generiamo dati da una t di studente con 1 gdl
plot(density(dat4), xlim=c(-6,6), ylim=c(0,0.6))
# sovrapponiamo la funzione di densità di una gaussiana e dovremmo notare differenze sulle code
curve(dnorm(x,0,1), col=2, lwd=2, add=TRUE)

set.seed(2222)
dat5<-rnorm(100) # generiamo dati da una gaussiana standard
# tracciamo il grafico dei quantili ordinando il vettore
dat5s<-sort(dat5)
qe<-((1:100)-0.5)/100
# e tracciamo il grafico dei quantili
plot(qe, dat5s)
# sovrapponiamo ora il grafico dei quantili di una gaussiana standard
curve(qnorm(x,0,1), lwd=2, add=TRUE)


plot(qnorm(qe), dat5s, main="grafico quantile-quantile per confronto con gaussiana")
abline(0,1)
# Ora utilizziamo direttamente la funzione qqnorm()
qqnorm(dat5s, main="grafico qqnorm per confronto con gaussiana")
qqline(dat5s) # aggiunge la linea per verificare il confronto