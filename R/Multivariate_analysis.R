library(MASS)
data(whiteside)
attach(whiteside)
House_prices = read.csv("./house_price.csv", sep = ",", stringsAsFactors=T, row.names=1)

plot(whiteside$Temp,whiteside$Gas)

codev = sum((whiteside$Temp-mean(whiteside$Temp))*(whiteside$Gas-mean(whiteside$Gas)))
dev = sum((whiteside$Temp-mean(whiteside$Temp))^2)
b1 = codev/dev
b0 = mean(whiteside$Gas) - b1 * mean(whiteside$Temp)

abline(b0,b1,col=2)

varres = sum((whiteside$Gas-(b0 + b1*whiteside$Temp))^2)

b1q = -(quantile(whiteside$Gas, probs = 0.75) - quantile(whiteside$Gas, probs = 0.25))/
  (quantile(whiteside$Temp, probs = 0.75) - quantile(whiteside$Temp, probs = 0.25))
b0q = mean(whiteside$Gas) - b1q * mean(whiteside$Temp)

abline(b0q,b1q, col= 1)

#' dev(y) = dev(residua) + dev(spiegata) -> sum((y_i - (y_i)^)^2) + sum(((y_i)^ - mean(y))^2)
#' var(y) = var(in) + var(fra) -> varianza nel boxplot + varianza delle medie dei boxplot
#' 0 <= dev(spiegata)/dev(y) = 1 - dev(residua)/dev(y) = R^2 <= 1 -> se 0 allora la retta è parallela, se 1 allora tutta sulla retta
#' R^2 = r_(x,y)^2
#' R^2 = coefficiente di determinazione lineare
regr = lm(Gas~Temp)
summary(regr)
table(b0,b1)
vec1 = rep(1, length(whiteside$Temp))
xx = cbind(vec1,whiteside$Temp)

bb = solve(t(xx)%*%xx)%*%t(xx)%*%whiteside$Gas
bb
detach(whiteside)


####


library(insuranceData)
data("AutoBi")
attach(AutoBi)

#' 1.) plot on loss and age
#' 2.) Analisi di regressione lineare (si puo utilizzare lm)

plot(CLMAGE,LOSS)
regr2 = lm(LOSS~CLMAGE)
summary(regr2)
codeva = sum((CLMAGE-mean(CLMAGE, na.rm = T))*(LOSS-mean(LOSS)),na.rm = T)
deva = sum((CLMAGE-mean(CLMAGE, na.rm = T))^2, na.rm = T)
b1a = codeva/deva
b0a = mean(LOSS)-b1a*mean(CLMAGE,na.rm = T)
table(beta0=b0a, beta1=b1a)
#abline(b0a,b1a, col = 2)
abline(lm(LOSS~CLMAGE))

plot(CLMAGE,log(LOSS))
regr3 = lm(log(LOSS)~CLMAGE)
summary(regr3)
abline(regr3$coefficients[1],regr3$coefficients[2], col = 3)
abline(lm(log(LOSS)~CLMAGE))

detach(AutoBi)


###


case = read.csv("case.csv")

attach(case)
plot(SquareFeet,Price)
abline(lm(Price~SquareFeet))
summary(lm(Price~SquareFeet))

lines(ksmooth(SquareFeet,Price, bandwidth = 1400), col=2, lty=4)
lines(ksmooth(SquareFeet,Price, bandwidth = median(SquareFeet)), col=3, lty=1, lwd =3)
lines(lowess(Price~SquareFeet, f=.8), col=4,lwd =3)
lines(lowess(Price~SquareFeet, f=.4), col=5,lwd =3)
lines(lowess(Price~SquareFeet, f=.2), col=6,lwd =3)

##
library(ggplot2)
ggplot(mapping = aes(x = SquareFeet, y = Price)) +
  geom_point() +
  geom_smooth(aes(), se=F)
library(datasets)
ggplot(mapping = aes(x = Sepal.Length, y = Petal.Length), data = iris) +
  geom_point() +
  geom_smooth(aes(colour=Species), se=F)
detach(case)
##

### Regrsione Lineare multiple
#' y un var quantitativa continua
#' y^(-)_i = M(y|x_i) media
#' V(y_i - y^(^)_i) varianza dei residui
#' \beta^(^) = (X^(T)X)^(-1)X^(T)y
#' residui(r_i) = (y_i - y^(^)_i)
#' sum((y_i - y^(^)_i)^2) = dev(residui)
#' dev(y) = dev(residui) + dev(spiegata) ->
#' dev(y) = sum((y_i - mean(y))^2)
#' dev(spiegata) = sum((y^(^)_i)^2 - mean(y))
#' Se aumento le variabili, R^2 aumentero
#' Smettero di aggiungere variabili se R^2 non aumenta di tanto
#' \beta^(^)_j/stderr(\beta^(^)_j) == N(0,1)-> guardo il two-tailed p-value
#' se piccolo lo prendo altrimenti lo scarto
#' Il grafico del r_i deve essere sparsa, se segue un andamento allora potrebbe richiedere
#' l'aggiunta di altre funzione (es: X^2 oppure log(X))
#' 
linm = lm(Gas~Temp)
res = residuals(linm)
yhat = fitted(linm)
summary(linm)
par(mfrow=c(4,4,2,1))
plot(yhat,res)
abline(h=0)

z = Insul
plot(Temp,Gas,pch=as.character(z),col=as.numeric(z)+1)
Insul = relevel(Insul, r="Before")
mlm = lm(Gas~Temp+Insul)
summary(mlm)
b0 = coef(mlm)[1]
b1 = coef(mlm)[2]
b2 = coef(mlm)[3]
abline(b0,b1,col=2)
abline(b0+b2,b1,col=3)

plot(mlm)


linmint = lm(Gas~Temp+Insul+Insul*Temp)
summary(linmint)
abline(linmint)
plot(linmint)
