### 1

v = c(rnorm(3,5,1))
m = matrix(nrow = 3, ncol = 5)
l = list("maria","luigi","comera")

lista = list(v,m,l)
names(lista) = c("rnorm","matrix","cantanti")
lista$rnorm

### 2

m = matrix(2*(0:5),nrow = 2, ncol = 3)
# Vengono aggiunti colonna per colonna
# Si puo cambiare a riga per riga mettendo byrow = T
is.matrix(m)
is.array(m)
# Ritorna True per entrambe is.matrix perchè matrice
# is.array perchè matrice e una forma n-dimensionale 
# di un array, dove n sono le colenne
b = m[,3]
b
# ritorna un vetorre numeric
b = as.matrix(b)

### 3

life = read.csv("Life.csv")
str(life)
# country_code = chr, qualitative, country_name = chr, qualitative
# year = int, qualitative, value =  numeric, quantitative, empty val = 9999
# Si, quelli qualitative
life = life[-1,,drop=F]
life$country_name = factor(life$country_name)
life$year = as.factor(life$year)
levels(life$country_name)
levels(life$year)
med_country = tapply(life$value, life$country_name, median)
med_country["Australia"]
as.matrix(med_country)

### 4

machinedata = read.table("macchine.data", sep = ",", na.strings = "?", stringsAsFactors = T)
str(machinedata)
# Machinedata è un dataframe formato da 26 colonne e 205 righe.
# Esso è composto da int, char e numeric sia qualitative che quantitative
two_and_more_NA = names(machinedata[colSums(is.na(machinedata))>=2])
# V3, perchè marca; V4, perchè tipo di benzina; V5, perchè tipo di motore
# V6, perchè la quantità di porte; V7, prechè tipo di auto; 
# V8, perchè il tipo di guida, V9; V15; V16; V18
# Tutti categoriali
str(machinedata)
levels(machinedata$V16) = c(8,5,4,6,3,12,2)
machinedata$V16 = as.numeric(as.character(machinedata$V16))
machinedata = machinedata[order(machinedata$V16,decreasing = TRUE),]
as.factor(machinedata$V16)

table1 = subset(machinedata, subset = V16>6, c(V3,V26))
table1$V3 = factor(table1$V3, levels = unique(table1$V3))
table2 = prop.table(table(machinedata$V16))
greatest_2_percintile = sort(table2, decreasing = T)[1:2]


macchineCil = subset(machinedata, subset = V16>6)


table3 = table(machinedata$V16)
prezzi_medi_al_cilindro = tapply(machinedata$V26, machinedata$V16, sum, na.rm=T)/table3

macchineCil$prezzoCat = cut(macchineCil$V26, 
                        breaks = c(0,10000,15000,20000,30000,max(machinedata$V26, na.rm = T)),
                        labels = c("fino a 10000", "da 10000 fino a 15000", "da 15000 a 20000", "da 20000 a 30000", "pi`u di 30000"))
































### Nuovo Pagina

fibo <- function(n){
  previous2 = 0
  previous1 = 0
  current = 1
  ratio = c()
  for (i in 2:n){
    previous2 = previous1;
    
    previous1 = current;
    
    current = previous2 + previous1;
    
    ratio[i-1] = current/previous1
  }
  fib_and_ratio = list(current,ratio)
  names(fib_and_ratio) = c("Fibonacci Number", "Ratio(n/n-1)")
  return(fib_and_ratio)
}

fibo(10)

### 2

x = runif(20,0,100)
x_tilde = c()
for (i in 1:length(x))
{
  if (x[i] > mean(x)){
    x_tilde[i] = x[i]
  }
  else{
    x_tilde[i] = mean(x)
  }
}
x_tilde

### 3

fun <- function(x){
  if (x < 0){
    result = x**2 - 1
  }
  else{
    result = x**3 - 1
  }
  return(result)
}
fun(-3)

### 4

rabbits = read.csv("Rabbits.csv")
rabbits = melt(rabbits, measure.vars = colnames(rabbits)[3:7])
temp = unique(rabbits[c("Treatment","Dose")])
rabbits = unstack(rabbits, value ~ variable)
rabbits = cbind(temp,rabbits)

### 5
set.seed(123)
iris2 = iris[sample(1:nrow(iris),sample(1:nrow(iris),1)),]
iris2$Sepal.Length = cut(iris2$Sepal.Length, breaks = 5)
freq_abs_iris2 = table(iris2$Sepal.Length)
freq_rel_iris2 =prop.table(table(iris2$Sepal.Length))
iris_copy = iris
iris_copy$Sepal.Length = cut(iris$Sepal.Length, breaks = 5)
freq_abs_iriscopy = table(iris_copy$Sepal.Length)
freq_rel_iriscopy = prop.table(table(iris_copy$Sepal.Length))

iris3_my_copy = subset(iris, subset = Species == "setosa", colnames(iris))































### Next Page

threshold <- function(x,n){
  return(sum(x>n, na.rm = T))
    
}

n = 50
x = sample(1:100,30)
threshold(x,n)
load("chicago_air.rda")
n = c(0.075, 90, 1.25)
threshold(chicago_air$ozone,n)
mapply(threshold, chicago_air[c("ozone","temp","solar")], n)

ceo = read.csv("Ceo.csv", sep = ";")
summary(ceo$Tot.Comp)
hist(ceo$Tot.Comp, xlab = "redditi Ceo")
abline(v = mean(ceo$Tot.Comp), col = "red", lwd = 2)
abline(v = min(ceo$Tot.Comp), col = "blue", lwd = 2)
abline(v = max(ceo$Tot.Comp), col = "green", lwd = 2)
abline(v = median(ceo$Tot.Comp), col = "orange", lwd = 2)

par(mfrow=c(1,1))
ecdf_ceo = ecdf(ceo$Tot.Comp)
x <- seq(min(ceo$Tot.Comp), max(ceo$Tot.Comp), length.out = 100)
y <- pnorm(x, mean = mean(ceo$Tot.Comp), sd = sd(ceo$Tot.Comp))

plot(ecdf_ceo, main = "ECDF vs Distribuzione Normale", xlab = "Tot.Comp", ylab = "Probabilità <= x", col = "blue")
lines(x, y, col = "red")


boxplot(log(ceo$Tot.Comp), ylim = c(log(min(ceo$Tot.Comp))-.5,log(max(ceo$Tot.Comp))+.5))


fun1 <- function(n,mu,s){
  sampple = rnorm(n,mu,s)
  hist(sampple, probability = T)
  lines(density(sampple))
}


fun1(30,10,1)

plot(c(0:100),c(0:100))
abline(v=mean(0:100), col = "red")
abline(h=mean(0:100), col = "blue")
abline(a=1, b=45, col = "yellow")
# 