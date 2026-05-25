mean <- function(x)
{
  total <- 0
  for(i in 1:length(x))
  {
    total = x[i] + total
  }
  return(total/length(x))
}




mean2 <- function(x)
{
  return(sum(x)/length(x))
}




pow <- function(x,y)
{
  return(x^y)
}




popvar <- function(x)
{
  total <- 0
  for(i in 1:length(x))
  {
    total = pow(x[i] - mean(x),2) + total
  }
  return(total/(length(x)))
}




popvar2 <- function(x)
{
  
  return(sum((x-mean(x))*(x+mean(x)))/(length(x)))
}




samplevar <- function(x)
{
  total <- 0
  for(i in 1:n)
  {
    total = pow(x[i] - mean(x),2) + total
  }
  return(total/(length(x)-1))
}




samplevar2 <- function(x)
{
  
  return(sum((x-mean(x))*(x+mean(x)))/(length(x)-1))
}