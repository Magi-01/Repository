library(MASS)


par(mfrow=c(1,2))
# consideriamo l'insieme di 9 dati nel vettore xx
xx<-c(148, 155, 162, 168, 172, 174, 175, 178, 186)
# rappresentiamo i punti su un grafico
# i parametri ylim e xlim permettono di definire
# i limiti per le coordinate x e y, pch permette di scegliere il simbolo
# (4 è il simbolo X), cex le dimensioni del simbolo.
nn<-length(xx)
plot(xx,rep(0,nn), ylim=c(0,1.1), xlim=c(135,200), pch=4, cex=1.5,
     main="esempio", xlab="h=10", ylab="proporzione")
abline(0,0)
# abline(a,b) traccia una retta con intercetta e coefficiente angolare
# con a e b assegnati
# points() permette di aggiungere al grafico esistente nuovi punti
points(c(0,xx),c(0,1:9)/nn, pch=19)
# e il parametro type="s" consente di ottenere una funzione a gradini
plot(xx,rep(0,nn), ylim=c(0,1.1), xlim=c(140,195), pch=4, cex=1.5,
     main="esempio", xlab="h=10", ylab="proporzione")
abline(0,0)
points(c(0,xx),c(0,1:9)/nn, type="s")
par(mfrow=c(2,2))
plot(ecdf(xx))


par(mfrow=c(2,2))
k = hist(Cars93$Length)
str(k)
k$density
hist(Cars93$Length, prob = T)
par(mfrow=c(1,1))


par(mfrow=c(2,2))
hist(Cars93$Length, main="istogramma ottenuto usando parametri di default")
hist(Cars93$Length, prob=TRUE, main="istogramma con le densità")
hist(Cars93$Length, prob=TRUE, breaks=12, main="istogramma 12 intervalli")
hist(Cars93$Length, prob=TRUE, breaks=c(140,160,170,180,190,200,220),
     main="istogramma con classi di diversa ampiezza")



data(geyser)
hist(geyser$waiting, prob=TRUE)
hist(geyser$waiting, breaks=c(43,(5*(1:14)+43)), prob=TRUE)


par(mfrow=c(1,2))
# versione della densità ottenuta applicando la definizione in (2)
nucl<-function(x,xx,h=1) {
  y=0
  n=length(xx)
  nucl=0
  for (i in 1:n){
    nucl=nucl+(abs(x-xx[i]) <= h/2)*1/(n*h)}
  nucl
}
plot(sort(Cars93$Length),nucl(sort(Cars93$Length),Cars93$Length,10),
     cex=.5, pch=19, ylab="Lenght", xlab="densità")
# versione del grafico in cui il calcolo viene fatto per tutti
# i punti di x e non solo in quelli osservati
plot(seq(130,230),nucl(seq(130,230),Cars93$Length,10), lwd=1.5,
     type="s", ylab="Lenght", xlab="densità")

par(mfrow=c(1,1))
den <- density(geyser$waiting)
plot(den, main="tempi di attesa fra successive eruzioni")


#excersice:
par(mfrow=c(2,2))
hist(Cars93$Length, prob = T)

dencar <- density(Cars93$Length)
str(dencar)
plot(dencar)

hist(Cars93$Length, prob = T)
lines(dencar)

par(mfrow=c(2,2))
dencar <- density(Cars93$Length, bw = 10)
#plot(dencar, 10)
hist(Cars93$Length, prob = T)
lines(dencar)
dencar <- density(Cars93$Length, bw = 1)
#plot(dencar,1)
hist(Cars93$Length, prob = T, ylim = c(0.000,0.035))
lines(dencar)
dencar <- density(Cars93$Length)
#plot(dencar)
hist(Cars93$Length, prob = T)
lines(dencar)
