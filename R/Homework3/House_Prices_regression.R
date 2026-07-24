HP = read.csv("house_price.csv", stringsAsFactors = T, row.names = 1)


library(reshape2)


HP = lapply(HP, as.numeric)


k = matrix(ncol = NCOL(HP))
for (var in names(HP)){
  k[var] = c(cor(HP$SalePrice,HP[[var]]))
}
k = melt(k)
names(k) = "SalePrice"
k

withNA = rownames(k)[is.na(k$SalePrice)]
withNA = withNA[-1]

for (ele in withNA){
  HP_lotfrontage = HP[[ele]][which(!is.na(HP[[ele]]))]
  HP_lotfrontageSALE = HP$SalePrice[which(!is.na(HP[[ele]]))]
  k[ele,"SalePrice"] = cor(HP_lotfrontageSALE,HP_lotfrontage)
}




library(ggplot2)
ggplot(data = k, aes(rownames(k),colnames(k))) + geom_tile(color = "white")+
  scale_fill_gradient2(low="red",high = "blue",mid="white",midpoint=0,limit=c(-1,1))+
  theme_minimal()

write.csv(k, file = "correlation.csv")

data("mtcars")
pairs(mtcars[,c("mpg", "disp", "hp", "wt")])
