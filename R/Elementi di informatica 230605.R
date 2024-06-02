#' ---
#' title: " Elementi di informatica - primo appello "
#' subtitle: AIDA
#' author: "Fadhla Mohamed Mutua - SM3201434"
#' date: "2024-04-17"
#' ---

# Il file 'nazioni.csv' contiene informazioni su 105 Nazioni e per ognuna di 
# esse riporta:
# - areaGeo: la regione geografica
# - reddito: il reddito pro capite in dollari
# - infmort: il tasso di mortalità infantile (morti ogni 100 nascite)
# - oil: se il paese esporta petrolio (1: no; 2: sì)

#' 1. [2] Caricare i dati nello spazio di lavoro in un data frame chiamato "nazioni",
#' stampare il numero di righe dell'oggetto importato ed il nome delle variabili.
#' \

nazioni <- read.csv("nazioni.csv")
print(nrow(nazioni))
print(colnames(nazioni))

#' 2. [3] Stampare il vettore con il numero di valori mancanti presenti in ogni 
#' variabile.
#' \

numof_na <- as.vector(colSums(is.na(nazioni)))
print(numof_na)

#' 3. [2] Ottenere il nome dei Paesi in cui sono presenti valori mancanti. 
#' \

namof_na <- unique(nazioni$nome[apply(is.na(nazioni), 1, any)])
print(namof_na)

#' 4. [2] Eliminare i valori mancanti dal data set. 
#' \

nazioni_rmvna <- na.omit(nazioni)
print(nazioni_rmvna)

#' 5. [3] Ottenere la distribuzione delle frequenze percentuali della variabile 
#' 'areaGeo' ed ordinarle in ordine decrescente.
#' \

areaGeo <- nazioni_rmvna$areaGeo
nareaGeo <- table(areaGeo)
distribution <- prop.table(nareaGeo)*100
distribution_dec <- sort(distribution, decreasing = T)
print(distribution_dec)

#' 6. [4] Convertire la variabile areaGeo in factor ordinando i livelli secondo 
#' l'ordine ottenuto al punto precedente. Salvare il factor come nuova variabile
#' del data frame chiamata areaGeofact ed eliminare la variabile areaGeo.
#' \

areaGeofact <- factor(areaGeo, labels = c("Africa", "Asia", "America", "Europe"))
nazioni_rmvna$areaGeofact <- areaGeofact
nazioni_rmvna$areaGeo <- NULL

#' 7. [3] Convertire la variabile oil in factor utilizzando i livelli: "no" e "yes".
#' Sovrascrivere la variabile oil già presente nel data.frame.
#' \

foil <- factor(nazioni_rmvna$oil, labels = c("no", "yes"))
nazioni_rmvna$oil <- foil

#' 8. [3] Quali Paesi esportano petrolio e in quali regioni si trovano? 
#' Stampare il risultato in due colonne. 
#' \

expnazioni <- subset(nazioni_rmvna, subset = (oil == "yes"), select = c(nome, areaGeofact))
print(expnazioni)

#' 9. [3] Calcolare il tasso di mortalità infantile medio in ogni area geografica.
#' \

x <- tapply(nazioni_rmvna$infmort, nazioni_rmvna$areaGeofact, mean)
print(round(x,digits <- 2))

#' 10. [2] Quante nazioni hanno un tasso di mortalità infantile superiore o 
#' uguale a 300? 
#' \

infmort_300 <- sum(nazioni_rmvna$infmort>=300)
print(infmort_300)

#' 11. [2] Quante delle nazioni identificate al punto 10 esportano petrolio?
#' \

expinfmort_300 <- sum(nazioni_rmvna$infmort>=300 & nazioni_rmvna$oil=="yes")
print(expinfmort_300)

#' 12. [4] Dividere la finestra grafica in 2 righe e 2 colonne. In ogni spazio,
#' rappresentare con un boxplot la distribuzione della mortalità infantile 
#' condizionata alla regione geografica. Impostare lo stesso range sull'asse y
#' ed il titolo del grafico.
#' \

minimum = min(nazioni_rmvna$infmort)
maximum = max(nazioni_rmvna$infmort)

par(mfrow = c(2, 2))
for (continent in levels(nazioni_rmvna$areaGeofact))
{
  fivenum(nazioni_rmvna$infmort[which(nazioni_rmvna$areaGeofact == continent)])
  boxplot(nazioni_rmvna$infmort[which(nazioni_rmvna$areaGeofact == continent)] ~ 
          nazioni_rmvna$areaGeofact[which(nazioni_rmvna$areaGeofact == continent)],
          main = paste("infant mortality rate in",continent), xlab = continent,
          ylab = "infant mortality",
          ylim = c(minimum,maximum),
          drop = T)
}
mtext("Infant mortality Rate Based on Continent", line = 0.5, cex = 1.5, 
      side = 0, adj = 1.5, padj = 1)

par(mfrow = c(1, 1))

#' 13. [2] Rappresentare con un istogramma la distribuzione del reddito. 
#' Modificare l'etichetta dell'asse x con il nome della variabile 
#' ed eliminare il titolo.
#' \

hist(nazioni_rmvna$reddito, main = "", xlab = "reddito")

#' 14. [3] Aggiungere al grafico precedente le mediane del reddito per area 
#' geografica utilizzando dei punti di colore diverso.
#' \

med <- tapply(nazioni_rmvna$reddito, nazioni_rmvna$areaGeofact, median)
colors <- rainbow(nlevels(nazioni_rmvna$areaGeofact))
points(x = med, y = rep(10, length(med)), col = colors, pch = 16)
# Better Visualazation:
# mtext("Median : Africa - Asia - America - Europe", 
#       line = 0.5, cex = 1.5, side = 1, padj = 3.8)
# segments(x0 = med, y0 = 0, x1 = med, y1 = par("usr")[4], col = colors, lwd = 2)

#' 15. [2] Dividere la variabile reddito in classi utilizzando le seguenti 
#' categorie: "fino a 500", "(500, 1500]", "(1500, 4000]", "4000 e più".
#' Salvare la nuova variabile in un oggetto chiamato redditoCat.
#' \

redditoCat <- cut(nazioni_rmvna$reddito, breaks = c(0,500,1500,4000,Inf), 
                 labels = c("fino a 500", "(500, 1500]", "(1500, 4000]","4000 e più"))

#' 16. [2] Quante nazioni sono nella categoria "4000 e più"? E qual è la loro 
#' distribuzione per area geografica?
#' \
 
sumtotal_above_4000 <- sum(redditoCat=="4000 e più")
sumper_areageo <- table(nazioni_rmvna$areaGeofact[which(redditoCat == "4000 e più")])
print(sumtotal_above_4000)
print(sumper_areageo)

#' 17. [3] Stampare le distribuzioni condizionate della variabile redditoCat 
#' rispetto all'esportazione di petrolio approssimandole a 2 cifre decimali.
#' \

distr_reddito_expoil <- round(prop.table(table(redditoCat,nazioni_rmvna$oil), margin = 2),2)
print(distr_reddito_expoil)

