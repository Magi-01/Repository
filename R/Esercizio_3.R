dataset = matrix(1:50,nrow = 10,ncol = 5)

1:ncol(dataset)
minimum <- apply(dataset,2,min)
maximum <- apply(dataset,2,max)
quartile = matrix(nrow = ncol(dataset),ncol = 3)

for (j in 1:ncol(dataset)){
  quartile[j,] <- quantile(dataset[,j],probs=c(0.25,0.50,0.75))
}


quartile


calculated <- data.frame(minimum, maximum, quartile)

