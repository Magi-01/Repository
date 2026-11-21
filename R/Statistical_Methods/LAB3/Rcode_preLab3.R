#################################
# R code for LAB3 - 23/10/2026  #
#                               #
# ! This is the pre-LAB version #
# You must replace XXX properly #
#################################



#   _____________________________________________________________________________________
#   Interval estimation: for $\mu$ under the Gaussian case, when $\sigma^2$ is known ####

B <- 1000     # Number of replications
n <- 10       # sample size
mu <- 5       # True population mean 
sigma <- 2    # True population standard deviation

alpha <- 0.05 # Confidence level: 1- alpha

# CI is matrix where we save the confidence intervals for each replication:
# -) first column: lower bound
# -) second column: upper bound
CI <- matrix(0, B, 2)

# "l" is a vector whose elements assume TRUE (1) or FALSE(0) depending on 
# whether the true parameter value lies within the interval
l <- rep(0, B)

###############################################################################
# TODO:
# -) Generate samples from the N(mu, sigma^2)
# -) Obtain the confidence interval for each replication 
# -) Use a flag to detect if the confidence interval 
#    includes the true parameter value
set.seed(1234)
q0975 <- qnorm(1 - alpha/2) # quantile of order 1-alpha/2 of N(0,1)
for(i in 1 : B) {
  x <- rnorm(n, mu, sigma)
  CI[i,] <- mean(x) + c(-q0975, q0975) * sigma/sqrt(n)
    l[i] <- (mu > CI[i,1]) & (mu < CI[i,2])
}

#Empirical coverage probability:

#XXX
###############################################################################


#Plot the first 100 c.i.:
# black: intervals not including mu
# red: intervals including  mu
plot(1, xlim = c(0, 10), ylim = c(0, 11), type = "n", 
     xlab = expression(mu), ylab = "", yaxt  = "n", 
     main = paste("100 IC for the mean (known variance)"), cex.main = 1.2)
abline(v = mu)

d <- 0
for(i in 1 : 100){
  d <- d + 0.1 
  lines(seq(CI[i, 1], CI[i, 2], length = 100), rep(d, 100), col = (l[i] + 1))
}

# number of intervals (out the 100) including the true parameter value 
sum(l[1 : 100]) 


#   _______________________________________________________________________________________
#   Interval estimation: for $\mu$ under the Gaussian case, when $\sigma^2$ is unknown ####

# consider the same setting as above: 
# mu = 5, sigma=2, n=10, B=1000, alpha=0.05
CI <- matrix(0, B, 2)
l <- rep(0, B)

set.seed(1234)
q0975 <- qt(1 - alpha/2, n - 1) # quantile of order 1-alpha/2  of t(n-1)

###############################################################################
# TODO:
# -) Generate samples from the N(mu, sigma^2)
# -) Obtain the confidence interval for each replication 
# -) Use a flag to detect if the confidence interval 
#    includes the true parameter value
for(i in 1 : B) {
  x <- rnorm(n, mu, sigma)
  CI[i,] <- mean(x) + c(-q0975, q0975) * sd(x)/sqrt(n)
  l[i] <- (mu > CI[i,1]) & (mu < CI[i,2])
}

#Empirical coverage probability:

#XXX
###############################################################################


# Plot of the (first 100) CIs
plot(1, xlim = c(0,10), ylim = c(0,11), type = "n", 
     xlab = expression(mu), ylab = "", yaxt  = "n", 
     main = paste("100 IC for the mean (unknown variance)"), cex.main = 1.2)
abline(v = mu)

d <- 0
for (i in 1 : 100){
  d <- d + 0.1 
  lines(seq(CI[i, 1], CI[i, 2], length = 100), rep(d, 100), col = (l[i] + 1))
}

# number of intervals (out the 100) including the true parameter value 
sum(l[1 : 100]) 





#   ____________________________________________________________________________
#   Interval estimation: difference between two means                       ####

Anor <- read.table("http://stat4ds.rwth-aachen.de/data/Anorexia.dat", 
                   header=TRUE)
# Get difference post-pre treatment for the group cb and c 
cogbehav <- Anor$after[Anor$therapy == "cb"] - Anor$before[Anor$therapy == "cb"]
control <- Anor$after[Anor$therapy == "c"] - Anor$before[Anor$therapy == "c"]


# Get the 95% CI via t.test function   
res <- t.test(cogbehav,control,var.equal=TRUE,conf.level=0.95)
res$conf.int

###############################################################################
# TODO:
# Obtain the result above by hand
# Fill XXX
n1 <- length(cogbehav)
n2 <- length(control)

s2 <- ((n1-1)*var(cogbehav) + (n2-1)*var(control))/(n1+n2-2)

CI <- mean(cogbehav) - mean(control) + c(-1,1) * qt(0.975,n1+n2-2) * sqrt(s2*((1/n1) + (1/n2)))
###############################################################################


 
#   ____________________________________________________________________________
#   Interval estimation: difference between two proportions                 ####

success <- c(315, 304)
total <-  c(604, 597)
res <- prop.test(success, total, conf.level = 0.95, correct = FALSE)
res$conf.int

# By hand
p1 <- success[1]/total[1]
p2 <- success[2]/total[2]

p1 - p2 + c(-1, 1) * qnorm(0.975) * sqrt(p1 * (1 - p1)/total[1] + p2 * (1 - p2)/total[2])



#   ____________________________________________________________________________
#   Hypothesis testing                                                      ####


##  ............................................................................
##  Test for the mean difference                                            ####


# Two-sided two-sample test
# H0: mu1 = mu2
# H1: mu1 != mu2

res.two  <- t.test(cogbehav, control, var.equal = TRUE)
res.two

# PErform th test by hand (complete XXX)
testStat <- (mean(cogbehav) - mean(control))/sqrt(s2 * (1/n1 + 1/n2))

# p-value
p.value.two <- 2 * pt(testStat, df = n1 + n2 -2, lower.tail =  FALSE)

2 * (1 - pt(testStat, df = n1 + n2 -2))

# One-sided two-sample test
# H0: mu1 > mu2
# H1: mu1 <= mu2
res.one  <- t.test(cogbehav, control, var.equal = TRUE, alternative = "greater")
res.one

## one-sided
p.value.one <- pt(testStat, df = n1 + n2 -2, lower.tail =  FALSE)


library(RColorBrewer)
plotclr <- brewer.pal(6, "YlOrRd")

curve(dt(x, n1 + n2 - 2), xlim = c(-5, 5), ylim = c(0, 0.4), 
      main = "p-values and rejection region", col = "blue", 
      lwd = 2, xlab = "x-y",  ylab = bquote(t[.(n1 + n2 - 2)]),  yaxs="i")
cord.x <- c(qt(0.95, n1 + n2 - 2),seq(qt(0.95, n1 + n2 - 2), 5, 0.01), 5)
cord.y <- c(0, dt(seq(qt(0.95, n1 + n2 - 2), 5, 0.01), n1 + n2 - 2), 0)
polygon(cord.x, cord.y, col = plotclr[3], border = NA )

abline(v = res.one$statistic, lty = 2, lwd = 2, col = "red")
text(0, 0.2, paste("Accept", expression(H0)))
text(2.7, 0.08, paste("Reject", expression(H0)))
text(as.double(res.one$statistic) - 0.15, 0.02, "t", col = "red", cex = 1.2)
# On the plot above note that the value of observed test statistic can vary as
# function of the sample (only for this particular case is too close to alpha
# that you cannot detect a graphical difference)



##  ............................................................................
##  Test for equality of variance                                           ####

# Using the var.test function
var.test(cogbehav, control, alternative = "two.sided") 

# Test for equality of variance by hand
ratiovar <- var(cogbehav)/var(control) # Test statistic 
pv_bi <- 2 * min(pf(ratiovar, n1 - 1, n2 - 1, lower = FALSE), 
                 1 - pf(ratiovar, n1 - 1, n2 - 1, lower = FALSE))
pv_bi


##  ............................................................................
##  Pearson's chi-squared test                                              ####

## Test for independence
n <- 760
obs_freq <- matrix(c(50, 70, 30, 100,
                     114, 30, 10, 100, 
                     116, 27, 13, 100),3,4,T)
colnames(obs_freq) <- c("A", "B", "AB", "O")
rownames(obs_freq) <- c("South", "Central", "North")
chisq.test(obs_freq)


mx <- colSums(obs_freq)/n; mx
my <- rowSums(obs_freq)/n; my

# !!!!: Note that the result of outer()
# can be obtained by doing 
# t(mx %*% t(my))
# my%*%t(mx)
# tcrossprod(my,mx)

exp_freq <- outer(my, mx) * n; exp_freq
chi2 <- sum((obs_freq - exp_freq)^2/exp_freq); chi2
chi2
pchisq(chi2, (ncol(obs_freq) - 1) * (nrow(obs_freq) - 1), lower.tail = FALSE)

## Goodness of fit test

child <- 0:5
fam <- c(52, 60, 55, 18, 8, 7)
lambda <-  1.5
plot(0:5, fam/200, ylim = c(0,0.35))
segments(0:5,rep(0,5), 0:5, 
         c(dpois(0:4, lambda),ppois(4, lambda, lower = FALSE)))

c(dpois(0:4, lambda), ppois(4, lambda, lower = FALSE))


exp <- c(dpois(0:4, lambda),ppois(4, lambda, lower = FALSE)) * 200 
round(exp, 4)
chisq_el <- (fam-exp)^2/exp
round((fam-exp)^2/exp, 4)


chisq.obs <- sum(chisq_el)
chisq.obs 
pchisq(chisq.obs, df = 5, lower=FALSE)


chisq.test(fam, p = exp/200)

