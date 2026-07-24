---
title: "Homework 2"
author: "Mutua Fadhla Mohamed"
output:
  html_document: default
  pdf_document: default
---

# Quantitativo

## Impostazioni progetto

```{r, warning = F}
library(ggplot2)

plot_theme = theme_minimal() + 
    theme(
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.line = element_line(color = "darkgrey", size = 0.4),
        axis.ticks = element_line(color = "black"))
theme_set(plot_theme)
```

## Lettura dati

```{r, warning = F}
House_prices = read.csv2("./house_price.csv", sep = ",", stringsAsFactors=T, row.names=1)
```

## Scelta di colonne

```{r, warning = F}
column_selection <- function(data){
  House_prices_numeric_bool = sapply(data, is.numeric)
  House_prices_numeric_fcolnames = names(data[House_prices_numeric_bool])
  # Rimozione delle colonne
  House_prices_numeric_fcolnames = House_prices_numeric_fcolnames[c(-1,-4,-5)]
  print("Colonne scelte: ")
  print(House_prices_numeric_fcolnames)
  return(House_prices_numeric_fcolnames)
}
House_prices_numeric_colnames = column_selection(House_prices)
```

## Analisi Primaria

```{r, warning = F}
Analisi_Primaria <- function(data){
  print(paste("Numeri di osservazione: ",nrow(House_prices)))
  print(paste("Numeri di variabili: ",length(data)))
}
Analisi_Primaria(House_prices_numeric_colnames)
```

## Analisi Univariata

```{r, warning = F}
univariate_analysis <- function(data, colname) {
  result = list()
  
  result[[colname]] = list(
    # Summary statistics including min, 1st and 3rd quartile, mean, median, max
    Summary = summary(data),
    # Variance
    Variance = round(var(data), 2),
    # Standard Deviation
    Standard_Deviation = round(sd(data), 2),
    # Mode calculation
    Moda = Mode(data)
  )
  return(result)
}

Mode <- function(data) {
  uniq_vals = unique(data)
  tab = tabulate(match(data, uniq_vals))
  modes = uniq_vals[tab == max(tab)]
  list(Mode = modes, Frequency = max(tab))
}

sum_info_h = lapply(House_prices_numeric_colnames, function(ele) {
  colummn = na.omit(House_prices[, ele])
  univariate_analysis(colummn, ele)
})
```



## Funzione di Visualizazione continuo

```{r, warning = F}
visualization_continuos <- function(data,sumary,colname){
  par(mfrow = c(2,2))
  print(sumary)
  boxplot(data, horizontal = T, main = paste("Boxplot of ", colname))
  hist(data, probability = F, col = 2, main = paste("Hist of ", colname))
  curve(dnorm(x, mean = mean(data), sd = sd(data)),
        add=T, col = 4)
  print(density(data))
  plot(density(data), main = paste("Desnity Distribution of ", colname))
  qqnorm(data, col = 3, main = paste("QQnorm of ", colname))
  qqline(data, col = 5, pch = 10)
}
```

## Funzione di Visualizazione discreta

```{r, warning = F}
visualization_discrete <- function(data,sumary,colname){
  par(mfrow = c(1,1))
  print(sumary)
  print("freq: ")
  cat("\n")
  print(table(data))
  cat("\n")
  print("percintile: ")
  cat("\n")
  print(prop.table(table(data))*100)
  cat("\n")
  barplot(table(data), col = 2, main = paste("Barplot of ", colname))
}
```

## Rimozione 0

```{r, warning = F}
rm_0 <- function(data){
  imp_data = data
  imp_data[imp_data == 0] = NA
  imp_data = na.omit(imp_data)
  return(imp_data)
}
```

## Rimozione degli outlier attraverso intervallo di confidenza

```{r, warning = F}
rm_oultiers <- function(data, CI){
  mean_data = mean(data)
  sd_data = sd(data)
  confidence_level = CI
  z_score = qnorm((1 + confidence_level) / 2)
  
  lower_bound = mean_data - z_score * sd_data
  upper_bound = mean_data + z_score * sd_data
  
  data_in_ci = data[data >= lower_bound & data <= upper_bound]
  return(data_in_ci)
}
```


## 1: Visualizazione LotFrontage
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"LotFrontage"]), sum_info_h[1],"LotFrontage")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte degli lotti ha un percorso verso la strada 
compreso tra 59 e 80 piedi, essendo rispettivamente il 1° e il 3° quantile e maggiormente
alineati alla retta qq della norma.
Rimangono ancora degli outlier
  
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(na.omit(House_prices[,"LotFrontage"]),.90)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"LotFrontage"),"LotFrontage")
```
Possiamo osservare che la distribuzione segue per lo piu quella gaussiana
e che la maggior parte degli lotti ha un percorso verso la strada 
compreso tra 60 e 80 piedi, essendo rispettivamente il 1° e il 3° quantile e con maggior
densita alineati alla retta qq della norma.


### 2: Visualizazione LotArea
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"LotArea"]), sum_info_h[2],"LotArea")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte degli lotti ha un area compreso 
tra 7554 e 11602 piedi^2, essendo rispettivamente il 1° e il 3° quantile e 
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier

## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(na.omit(House_prices[,"LotArea"]),.50)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"LotArea"),"LotArea")
```
Possiamo dedurre che la distribuzione è  positivamente asimetrica
e che la maggior parte degli lotti ha un area compreso 
tra 7500 e 11362 piedi^2, essendo rispettivamente il 1° e il 3° quantile e con maggior
densita alineati alla retta qq della norma.
Da notare che si butts il 50% dei dat


### 3: Visualizazione YearBuilt
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"YearBuilt"]), sum_info_h[3],"YearBuilt")
```
Possiamo dedurre che la distribuzione è negativamente asimetrica
e quindi il numero di edifici costruiti crescono ogni anno
La maggior parte degli edifici erano costruiti fra 1954 e 2000,
essendo rispettivamente il 1° e il 3° quantile e l'anno con
è il massimo costruitto è nel 2006 con 67 edifici


### 4: Visualizazione YearRemodAd
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"YearRemodAdd"]), sum_info_h[4], "YearBuilt")
```
Possiamo notare che la distribuzione ha un pico negli anni 50 con il massimo di rimodellazione
nel 1950 con 178 edifici. Dopodichè, si ha un calo quasi immediato
e i numeri di edifici rimodellati resta quasi costante fino agli anni 80
dove fra il 1980 al 1985 si nota il minimo numero di rimodellazione. Dopodiche,
la distribuzione inizia a crescere quasi costantemente con pico negli anni fra 2000 e 2005


### 5: Visualizazione MasVnrArea
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"MasVnrArea"]), sum_info_h[5],"MasVnrArea")
```
Possiamo dedurre che la distribuzione è  positivamente asimetrica.
Si puo osservare che l'area di MasVnrArea è prelevantemete 0 qundi non applicato.
Per fare un analisi sugli edifici che hanno il Masonry veneer, serve ignorare quelli senza
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"MasVnrArea"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"MasVnrArea"),"MasVnrArea")
```
Possiamo dedurre che la distribuzione è  positivamente asimetrica
e che la maggior parte degli edifici ha un area coperto
da Masonry veneer tra 113.0 e 330.5 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"MasVnrArea"),"MasVnrArea")
```
Possiamo dedurre nella distribuzione che la maggior parte degli edifici ha un area coperto
da Masonry veneer tra 108.5 e 304.0 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 6: Visualizazione BsmtFinSF1
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"BsmtFinSF1"]), sum_info_h[6],"BsmtFinSF1")
```
Possiamo dedurre che la distribuzione è  positivamente asimetrica.
Si puo osservare che l'area finito di semiterrato finito di tipo 1 è prelevantemete 0
qundi non costruitto.
Per fare un analisi sugli edifici che hanno il semiterrato finito di tipo 1, 
serve ignorare quelli senza
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"BsmtFinSF1"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"BsmtFinSF1"),"BsmtFinSF1")
```
Possiamo dedurre che la distribuzione è  positivamente asimetrica
e che la maggior parte degli edifici ha un semiterrato finito di tipo 1 di area compresa
tra 371.0 e 867.0 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"BsmtFinSF1"),"BsmtFinSF1")
```
Possiamo dedurre nella distribuzione semi-normato che la maggior parte degli edifici
ha un semiterrato finito di tipo 1 di area compresa tra 364.8 e 833.8 piedi^2,
essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 7: Visualizazione BsmtFinSF2
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"BsmtFinSF2"]), sum_info_h[7],"BsmtFinSF2")
```
Possiamo dedurre che la distribuzione è  positivamente asimetrica.
Si puo osservare che l'area finito di semiterrato finito di tipo 2 è prelevantemete 0
qundi non costruitto.
Per fare un analisi sugli edifici che hanno il semiterrato finito di tipo 2, 
serve ignorare quelli senza
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"BsmtFinSF2"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"BsmtFinSF2"),"BsmtFinSF2")
```
Possiamo dedurre che la distribuzione è  positivamente asimetrica
e che la maggior parte degli edifici ha un semiterrato finito di tipo 2 di area compresa
tra 178.5 e 551.0 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"BsmtFinSF2"),"BsmtFinSF2")
```
Possiamo dedurre nella distribuzione semi-normato che la maggior parte degli edifici
ha un semiterrato finito di tipo 2 di area compresa tra 173.8 e 512.2 piedi^2,
essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 8: Visualizazione BsmtUnfSF
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"BsmtUnfSF"]), sum_info_h[8],"BsmtUnfSF")
```
Possiamo dedurre che la distribuzione è  positivamente asimetrica.
Si puo osservare che l'area di semiterrato non finito è prelevantemete 0
qundi non costruitto.
Per fare un analisi sugli edifici che hanno il semiterrato non finito, 
serve ignorare quelli con
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"BsmtUnfSF"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"BsmtUnfSF"),"BsmtUnfSF")
```
Possiamo dedurre che la distribuzione è  positivamente asimetrica
e che la maggior parte degli edifici ha un semiterrato non finito di area compresa
tra 288.0 e 843.2 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"BsmtUnfSF"),"BsmtUnfSF")
```
Possiamo dedurre nella distribuzione semi-normato che la maggior parte degli edifici
ha un semiterrato non finito di area compresa tra 278.5 e 780.0 piedi^2,
essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 9: Visualizazione TotalBsmtSF
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"TotalBsmtSF"]), sum_info_h[9],"TotalBsmtSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica.
Si puo osservare che l'area totale semiterrato è prelevantemete 0
qundi non costruitto.
Per fare un analisi sugli edifici che hanno il semiterrato non finito, 
serve ignorare quelli senza
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"TotalBsmtSF"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"TotalBsmtSF"),"TotalBsmtSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte degli edifici ha un semiterrato di area compresa
tra 810.5 e 1309.5 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"TotalBsmtSF"),"TotalBsmtSF")
```
Possiamo notare che la maggior parte degli edifici ha un semiterrato di area compresa
tra 804.0 e 1266.8 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
  

### 10: Visualizazione X1stFlrSF
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"X1stFlrSF"]), sum_info_h[10],"X1stFlrSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte degli edifici ha il primo piano di area compresa
tra 882 e 1391 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(House_prices[,"X1stFlrSF"],.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"X1stFlrSF"),"X1stFlrSF")
```
Possiamo notare che la maggior parte degli edifici ha il primo piano di area compresa
tra 874 e 1344 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 11: Visualizazione X2ndFlrSF
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"X2ndFlrSF"]), sum_info_h[11],"X2ndFlrSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica.
Si puo osservare che l'area del secondo piano è prelevantemete 0
qundi non costruitto.
Per fare un analisi sugli edifici che hanno il  secondo piano, 
serve ignorare quelli senza
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"X2ndFlrSF"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"X2ndFlrSF"),"X2ndFlrSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte degli edifici ha il secondo piano di area compresa
tra 625.0 e 926.5 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.90)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"X2ndFlrSF"),"X2ndFlrSF")
```
Possiamo notare che la maggior parte degli edifici ha il secondo piano di area compresa
tra 634 e 895 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 12: Visualizazione LowQualFinSF
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"LowQualFinSF"]), sum_info_h[12],"LowQualFinSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica.
Si puo osservare che l'area dei piani fatti male è prelevantemete 0
Per fare un analisi sugli edifici che hanno i piani fatti male, 
serve ignorare quelli senza
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"LowQualFinSF"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"LowQualFinSF"),"LowQualFinSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte degli edifici ha i piani fatti male di area compresa
tra 168.2 e 477.5 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Gli edifici che hanno i piani fatti male sono cosi piccoli (rispetto al totale ~=1,8%)
che si puo trascurare


### 13: Visualizazione GrLivArea
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"GrLivArea"]), sum_info_h[13],"GrLivArea")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte degli edifici che sono sopra il seminterrato
sono di area compresa tra 1130 e 1777 piedi^2, essendo rispettivamente 
il 1° e il 3° quantile e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(na.omit(House_prices[,"GrLivArea"]),.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"GrLivArea"),"GrLivArea")
```
Possiamo notare che la maggior parte degli edifici che sono sopra il seminterrato
sono compresi tra 1121 e 1728 piedi^2, essendo rispettivamente
il 1° e il 3° quantile e con maggior densita alineati alla retta qq della norma.


### 14: Visualizazione BsmtFullBath
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"BsmtFullBath"]), sum_info_h[14],"BsmtFullBath")
```
# Descrete
Si osserva che la maggior parte dei semiterrati non hanno un bagno completo essendo
il ~=58.6% con frequenza di 856
Si osserva che la minima freq dei numeri di bagno completi e 1 per 3 bangi 
e quindi si puo trascurare essendo <1%
Si osserva che la freq dei numeri di bango completi per 2 bangi e 15 che è ~= 1.03% e 
quindi anch'esso trascurabile
Infine si osserva che la maggior parte dei semiterrati che hanno 1 bagno completo 
è ~=40.3% con frequenza di 588


### 15: Visualizazione BsmtHalfBath
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"BsmtHalfBath"]), sum_info_h[15],"BsmtHalfBath")
```
# Descrete
Si osserva che la maggior parte dei semiterrati non hanno un bagno mezzo completo essendo
il ~94% con frequenza di 1378
Si osserva che la freq dei numeri di bangi mezzi completi per 2 bangi e 2 che 
è ~= 1% e quindi anch'esso trascurabile
Infine si osserva che la maggior parte dei semiterrati che hanno 1 bagno mezzo completo 
è ~5% con frequenza di 80


### 16: Visualizazione FullBath
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"FullBath"]), sum_info_h[16],"FullBath")
```
# Descrete
Si osserva che la maggior parte degli edifici che sono sopra il seminterrato
hanno 2 bagni completi essendo il ~=52.6% con frequenza di 768
Si osserva che la minima freq dei numeri di bagno completi e 9 per nessun bango
e quindi si puo trascurare essendo <1%
Si osserva che la freq dei numeri di bango completi per 3 bangi e 33 che è ~= 2.2% e 
quindi anch'esso trascurabile
Infine si osserva che gli edifici che sono sopra il seminterrato
che hanno 1 bagno completo sono il ~=44.5% con frequenza di 650


### 17: Visualizazione HalfBath
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"HalfBath"]), sum_info_h[17],"HalfBath")
```
# Descrete
Si osserva che la maggior parte degli edifici che sono sopra il seminterrato
hanno 0 bagni mezzo completi essendo il ~=62.5% con frequenza di 913
Si osserva che la minima freq dei numeri di bagno completi e 12 per 2 bango
e quindi si puo trascurare essendo <1%
Infine si osserva che gli edifici che sono sopra il seminterrato
che hanno 1 bagno completo sono il ~=36.6% con frequenza di 535


### 18: Visualizazione BedroomAbvGr
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"BedroomAbvGr"]), sum_info_h[18],"BedroomAbvGr")
```
# Descrete
Si osserva che la maggior parte degli edifici che sono sopra il seminterrato
hanno 3 camere essendo il ~=55.07% con frequenza di 804 seguito da 2, 4 e 1 camere
essendo ~=24.5%, ~=14.6% e ~=3.4% con frequenze 358, 213 e 50 rispettivamente.
Si osserva che la minima freq dei numeri di camere e 1 per 8 camere seguito da
5, 6 e 0 camere con frequenze 21, 7 e 6 rispettivamente 
e quindi si puo trascurare essendo tutti <1.5%


### 19: Visualizazione KitchenAbvGr
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"KitchenAbvGr"]), sum_info_h[19],"KitchenAbvGr")
```
# Descrete
Si osserva che la maggior parte degli edifici che sono sopra il seminterrato
hanno 1 cucina essendo il ~=95.34% con frequenza di 1392 seguito da 2 cucine
essendo ~=4.45% con frequenze 65.
Si osserva che la minima freq dei numeri di camere e 1 per 0 cucine seguito da
3 cucine con frequenze 2 e quindi si puo trascurare essendo tutti <1%


### 20: Visualizazione TotRmsAbvGrd
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"TotRmsAbvGrd"]), sum_info_h[20],"TotRmsAbvGrd")
```
# Descrete
Si osserva che la maggior parte degli edifici che sono sopra il seminterrato
sono nel intervallo fra 5 e 7 stanze (non considerando i bagni) essendo il ~=68.8%
con il piu grande di occorenze 6 stanze (~=27.5%) con frequenza di 402.
Si osserva che la minima freq dei numeri di stanze e 1 per 2 e 14 stanze seguito da
12, 3 e 11 stanze con frequenze 11, 17 e 18 rispettivamente e quindi si puo 
trascurare essendo tutti <1.5%


### 21: Visualizazione Fireplaces
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"Fireplaces"]), sum_info_h[21],"Fireplaces")
```
# Descrete
Si osserva che la maggior parte degli edifici hanno 0 camini 
essendo il ~=47.26% con frequenza di 690 seguito da 1 e 2 camini 
essendo ~=44.52% con frequenze 650 e ~=7.88% con frequenza 115 rispettivamente.
Si osserva che la minima freq dei numeri di camini e 5 per 3 camini 
e quindi si puo trascurare essendo <1%


### 22: Visualizazione GarageYrBlt
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"GarageYrBlt"]), sum_info_h[22],"GarageYrBlt")
```
# Descrete
Possiamo dedurre che la distribuzione è positivamente asimetrica
e quindi il numero di garage costruiti crescono ogni anno
La maggior parte degli garage erano costruiti fra 1961 e 2002,
essendo rispettivamente il 1° e il 3° quantile e l'anno con
è il massimo costruitto è nel 2005 con 65 garage


### 23: Visualizazione GarageCars
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"GarageCars"]), sum_info_h[23],"GarageCars")
```
# Descrete
Si osserva che la maggior parte degli Garage possono avere 2 auto 
essendo il ~=56.44% con frequenza di 824 seguito da 1, 3 e 0 auto 
essendo ~=25.27% con frequenze 369, ~=12.4% con frequenze 181 
e ~=5.55% con frequenze 81 rispettivamente.
Si osserva che la minima freq dei numeri di auto e 5 per 4 auto 
e quindi si puo trascurare essendo <1%


### 24: Visualizazione GarageArea
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"GarageArea"]), sum_info_h[24],"GarageArea")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte degli garage sono di area compresa tra 334.5 e 576.0 piedi^2,
essendo rispettivamente il 1° e il 3° quantile e 
con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(na.omit(House_prices[,"GarageArea"]),.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"GarageArea"),"GarageArea")
```
Possiamo notare che la maggior parte degli garage
sono compresi tra 358.0 e 566.0 piedi^2, essendo rispettivamente
il 1° e il 3° quantile e con maggior densita alineati alla retta qq della norma.


### 25: Visualizazione WoodDeckSF
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"WoodDeckSF"]), sum_info_h[25],"WoodDeckSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica.
Si puo osservare che l'area del ponte di legno è prelevantemete 0 con 761 occorrenze
qundi non costruitto.
Per fare un analisi sui ponti di legno costruiti, serve ignorare quelli non
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"WoodDeckSF"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"WoodDeckSF"),"WoodDeckSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte dei ponte di legno sono di area compresa
tra 120.0 e 240.0 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"GarageArea"),"GarageArea")
```
Possiamo notare che la maggior parte dei ponte di legno sono di area compresa
tra 120.0 e 220.5 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 26: Visualizazione OpenPorchSF
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"OpenPorchSF"]), sum_info_h[26],"OpenPorchSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica.
Si puo osservare che l'area del portico aperto è prelevantemete 0 con 656 occorrenze
qundi non costruitto.
Per fare un analisi sui portico aperti, serve ignorare quelli non
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"OpenPorchSF"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"OpenPorchSF"),"OpenPorchSF")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte dei portico aperti sono di area compresa
tra 39.00 e 112.00 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"OpenPorchSF"),"OpenPorchSF")
```
Possiamo notare che la maggior parte dei portico aperti sono di area compresa
tra 120.0 e 220.5 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 27: Visualizazione EnclosedPorch
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"EnclosedPorch"]), sum_info_h[27],"EnclosedPorch")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica.
Si puo osservare che l'area del portico chiuso è prelevantemete 0 con 1252 occorrenze
qundi non costruitto.
Per fare un analisi sui portico chiusi, serve ignorare quelli non
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"EnclosedPorch"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"EnclosedPorch"),"EnclosedPorch")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte dei portico chiusi sono di area compresa
tra 104.2 e 205.0 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"EnclosedPorch"),"EnclosedPorch")
```
Possiamo notare che la maggior parte dei portico chiusi sono di area compresa
tra 102.0 e 200.5 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 28: Visualizazione X3SsnPorch
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"X3SsnPorch"]), sum_info_h[28],"X3SsnPorch")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica.
Si puo osservare che l'area del Portico per tre stagioni è prelevantemete 
0 con 1252 occorrenze qundi non costruitto.
Per fare un analisi sui Portico per tre stagioni, serve ignorare quelli non
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"X3SsnPorch"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"X3SsnPorch"),"X3SsnPorch")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte dei Portico per tre stagioni sono di area compresa
tra 150.8 e 239.8 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"X3SsnPorch"),"X3SsnPorch")
```
Possiamo notare che la maggior parte dei Portico per tre stagioni sono di area compresa
tra 146.2 e 216.0 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 29: Visualizazione ScreenPorch
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"ScreenPorch"]), sum_info_h[29],"ScreenPorch")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica.
Si puo osservare che l'area del Portico dello schermo è prelevantemete 
0 con 1344 occorrenze qundi non costruitto.
Per fare un analisi sui Portico dello schermo, serve ignorare quelli non
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"ScreenPorch"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"ScreenPorch"),"ScreenPorch")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte dei Portico dello schermo stagioni sono di area compresa
tra 143.8 e 224.0 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"ScreenPorch"),"ScreenPorch")
```
Possiamo notare che la maggior parte dei Portico dello schermo sono di area compresa
tra 142.2 e 214.5 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 30: Visualizazione PoolArea
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"PoolArea"]), sum_info_h[30],"PoolArea")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica.
Si puo osservare che l'area della piscina è prelevantemete 
0 con 1453 occorrenze qundi non costruitto.
Per fare un analisi sulle piscine, serve ignorare quelli non
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"PoolArea"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"PoolArea"),"PoolArea")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte delle piscine sono di area compresa
tra 515.5 e 612.0 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(rm_0_data,.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"PoolArea"),"PoolArea")
```
Possiamo notare che la maggior parte delle piscine sono di area compresa
tra 515.5 e 612.0 piedi^2, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.


### 31: Visualizazione MiscVal
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"MiscVal"]), sum_info_h[31],"MiscVal")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica.
Si puo osservare che il costo delle funzionalità varie è prelevantemete 
0 con 1453 occorrenze qundi non costruitto.
Per fare un analisi suli costi delle funzionalità varie, serve ignorare quelli non
## Rimozione 0
```{r, warning = F}
rm_0_data = rm_0(na.omit(House_prices[,"MiscVal"]))
visualization_continuos(rm_0_data, univariate_analysis(rm_0_data,"MiscVal"),"MiscVal")
```
Possiamo notare che la maggior parte dei costi delle funzionalità varie
sono di area compresa tra 515.5 e 612.0 piedi^2, 
essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Dato la variabilita dei dati e la mancanza di coerenza, dovuta alla mancanza di dati
(~=3.56%) dei dati totali (52 rispetto a 1460). Allora si potrebbe trascurare


### 32: Visualizazione MoSold
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"MoSold"]), sum_info_h[32],"MoSold")
```
# Descrete
Possiamo notare che i mesi con le moggior vendute sono fra il 5 e 8 mese con8 ~=55.69%
Dal 1 si ha una crescita di vendute fino al 6 mese dove si il massimo di vendite con 253
vendite dopodicche la vendita decresce 


### 33: Visualizazione YrSold
```{r, warning = F}
visualization_discrete(na.omit(House_prices[,"YrSold"]), sum_info_h[33],"YrSold")
```
# Descrete
Si puo notare che la vendita per anno dal 2006 a 2009 e per lo piu uniforme
con l'anno di maggior vendita nel 2009 con 338 venidite
Nel 2010 invece si ha il minimo di vendite con 175 vendite


### 34: Visualizazione SalePrice
```{r, warning = F}
visualization_continuos(na.omit(House_prices[,"SalePrice"]), sum_info_h[34],"SalePrice")
```
Possiamo dedurre che la distribuzione è positivamente asimetrica
e che la maggior parte del prezzo di vendita è compresa
tra 129975 e 214000, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
Rimangono ancora degli outlier
## Rimozione
```{r, warning = F}
rm_outlier_data = rm_oultiers(na.omit(House_prices[,"SalePrice"]),.95)
visualization_continuos(rm_outlier_data, univariate_analysis(rm_outlier_data,"PoolArea"),"PoolArea")
```
Possiamo notare che la maggior parte del prezzo di vendita sono compresa
tra 128988 e 196500, essendo rispettivamente il 1° e il 3° quantile
e con maggior densita alineati alla retta qq della norma.
