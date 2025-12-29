#' ---
#' title: "Basi di Dati e Data Analytics - prova intermedia "
#' subtitle: AIDA
#' author: "Fadhla Mohamed Muta - SM3201434"
#' date: "2023-04-18"
#' ---


#' 1. [2] Caricare i dati nello spazio di lavoro in un data frame dal nome "macchine"
#' e gestire i valori mancanti. Si noti che i valori mancanti sono codificati 
#' nel file con il carattere "?". ---------------
#' \

getwd()
machinedata = read.table("macchine.data", sep = ",", na.strings = "?")

#' 2. [1] Descrivere il tipo di oggetto importato e le sue dimensioni. ---------------
#' \

str(machinedata)
# il data frame è formato da 205 righe e 26 colonne senza nomi. Esso è una rappresentazione
# dei dati dei tipi di machine 

#' 3. [2] Quali variabili presentano più di 2 valori mancanti? ---------------
#' \
#' 
wherena <- c()
i = 1
for (colcheck in colnames(machinedata))
{
  if (sum(is.na(machinedata[,colcheck]))>=2)
  {
    wherena[i] = colcheck
    i = i+1
  }
}
wherena

#' 4. [2] Se presenti, le variabili importate come character si riferiscono a 
#' variabili di tipo categoriale? Scegliere ed applicare la conversione che si 
#' ritiene più opportuna. Fare riferimento al file info.names. ---------------
#' \

machinedata = data.frame(machinedata, stringsAsFactors = T)
str(machinedata)
machinedatatry = data.frame()

#' 5. [3] Si consideri la variabile V16 che indica il numero di cilindri delle vetture.
#' Ordinare i livelli della variabile secondo l'ordine crescente del numero di 
#' cilindri. ---------------
#' \

sort_machineadta_cilindri = machinedata[order(machinedata$V26),]

#' 6. [2] Quante macchine hanno più di 6 cilindri? ---------------
#' \
numeric_values = c(8,5,4,6,3,12,2)
table(machinedata$V16)

machinedata_fcilinder = factor(machinedata$V16, levels = c("eight", "five", "four", "six", "three", "twelve", "two"),
                              labels = numeric_values)
table_machinuedata_fcilinder = table(machinedata_fcilinder)

  
#' 7. [3] Quali marche (V3) e quali prezzi (V26) corrispond(ono alle macchine con due cilindri (V16)? ---------------
#' \

machinedata[which(as.numeric(as.character(machinedata_fcilinder)) == 2),c("V3","V26")]

#' 8. [2] Calcolare la distribuzione delle frequenze percentuali del numero di cilindri (V16)
#' approssimando i risultati ad una cifra decimale. ---------------
#' \

frequency = prop.table(table_machinuedata_fcilinder)

frequency = round(frequency, digits = 1)

#' 9. [3] Ottenere le due cilindrate maggiormente presenti nel dataset. ---------------
#' \

table_machinuedata_fcilinder = sort(table_machinuedata_fcilinder, decreasing = T)
two_most_present_cilinder = table_machinuedata_fcilinder[1:2]
names(two_most_present_cilinder)

#' 10. [3] Rappresentare graficamente la distribuzione delle frequenze relative della 
#' variabile V16 ed aggiungere al grafico il nome della variabile "Cilindri". ---------------
#' \
as.numeric(names(table_machinuedata_fcilinder))
plot(as.numeric(names(table_machinuedata_fcilinder)), frequency*10, main = "Cilindri", xlab = "cilinder", ylim = c(0,0.85)*10)

#' 11. [2] Creare un nuovo dataset chiamato "macchineCil" che contiene i dati relativi
#' alle sole macchine con il numero di cilindri trovati al punto 9. ---------------
#' \

vector_two_most_present_cilinder = as.vector(as.numeric(names(two_most_present_cilinder)))

temp = machinedata$V16
machinedata$V16 = machinedata_fcilinder
macchineCil = subset(machinedata, subset = (machinedata$V16 == vector_two_most_present_cilinder[1] |
                       machinedata$V16 == vector_two_most_present_cilinder[2]))
levels(macchineCil$V16) = c(4, 6)

machinedata$V16 = temp
macchineCil

#' 12. [3] Utilizzando il dataset "macchineCil", calcolare i prezzi medi (V26) rispetto al numero di cilindri (V16). ---------------
#' \


sum_eachprice = tapply(macchineCil$V26, macchineCil$V16, sum, na.rm=T)
ratio_eachprice = sum_eachprice / table(macchineCil$V16)


#' 13. [3] Utilizzando il dataset "macchineCil", a partire dalla variabile prezzo (V26), aggiungere una nuova variabile 
#' al dataframe chiamata "prezzoCat" considerando le seguenti categorie:
#' "fino a 10000", "(10000, 15000]", "(15000, 20000]", "(20000, 30000]", "30000 e più". ---------------
#' \

macchineCil$prezzoCat = cut(macchineCil$V26, 
                            breaks = c(0,10000,15000,20000,30000,max(macchineCil$V26,na.rm = T)),
                            labels = c("fino a 10000", "(10000, 15000]", "(15000, 20000]", "(20000, 30000]", "30000 e più"))

#' 14. [4] Utilizzando il dataset "macchineCil", rappresentare graficamente la variabile prezzoCat e aggiungere al grafico la media della
#' variabile prezzo (V26). ---------------
#' \
#' 
mean(macchineCil$V26, na.rm = T)
avrg_price = mean(macchineCil$V26, na.rm = T)

barplot(table(macchineCil$prezzoCat),
        main = "Distribuzione di prezzoCat",
        xlab = "prezzoCat",
        col = rainbow(length(levels(macchineCil$prezzoCat))),
        ylim = c(0,max(table(macchineCil$prezzoCat))+3))
points(avrg_price, type="l", lty=2, col=3, lwd=3)




#' 15. [4] Utilizzando il dataset "macchineCil", stampare in un unico oggetto le distribuzioni condizionate di prezzoCat 
#' rispetto  ai cilindri e la distribuzione marginale di prezzoCat. Commentare. ---------------
#' \

distr_prezzoCat = prop.table(table(macchineCil$V16, macchineCil$prezzoCat))
marg_distr_prezzoCat = margin.table(distr_prezzoCat, 1)
result = cbind(cond_distr = distr_prezzoCat, marg_distr = marg_distr_prezzoCat)
# Si ha che la marginal distribution funziona come row sum. Per ogni colonna, ad eccezione della marginale,
# appare la percentuale di costo rispetto al cilindro
result

#' 16. [6] Scrivere una funzione che prenda in input la numerosità del campione (n), 
#' la media (mu) e la deviazione standard (s) e:
#' - simuli un campione casuale di dimensione n da una normale con media mu e 
#'   deviazione standard s; 
#' - restituisca l'istogramma del campione simulato sovrapponendo la densità
#'   della distribuzione teorica. ---------------
#' \
funtt <- function(n,mu,s){
  x = rnorm(n,mean = mu, sd = s)
  hist(x, prob=T, xlab= "values")
  curve(dnorm(x, mean = mu, sd = s), col = "blue", lwd = 2, add = TRUE)
}

funtt(n = 30,mu = 10,s = 2)

