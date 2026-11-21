library(dplyr)

set.seed(42)
x <- seq(1e-6,0.999999,length.out=1000)
cols = c("red", "blue", "green", "yellow", "orange", "darkgreen")
alpha <- c(0.5,1, 10, 100)
beta <- alpha
# f <- function(x, a, b){
#   exp(lgamma(a+b) - (lgamma(a) + lgamma(b)) + log(x)*(a-1) + log(1-x)*(b-1))
# }

par(mfrow = c(1,3))

curve(dbeta(x,alpha[1],beta[1]), 1e-6, 999999e-6, col = cols[1], lwd = 2, ylim=c(0,12),
      xlab="x", ylab="f(x)",
      main=expression(paste("Beta pdf, ", alpha, " = ", beta)))
for (i in 2:4) curve(dbeta(x,alpha[i],beta[i]),add = TRUE, col = cols[i], lwd = 2)
legend("top", legend = paste0("a = b =", alpha), col = cols, lwd = 2)

ab = list(c(2,1), c(5,2), c(10,3), c(20,5), c(0.7,0.2), c(2,0.5))
curve(dbeta(x,ab[[1]][1],ab[[1]][2]), 1e-6, 999999e-6, col = cols[1], lwd = 2, ylim=c(0,6),
      xlab="x", ylab="f(x)",
      main=expression(paste("Beta pdf, ", alpha, " > ", beta)))
for (i in 2:6) curve(dbeta(x,ab[[i]][1],ab[[i]][2]),add = TRUE, col = cols[i], lwd = 2)
legend("top", legend = sapply(ab, function(p)paste0("a=",p[1]," b=", p[2])), col = cols, lwd = 2)

ab = list(c(1,2), c(2,5), c(3,10), c(5,20), c(0.2, 0.7), c(0.5,2))
curve(dbeta(x,ab[[1]][1],ab[[1]][2]), 1e-6, 999999e-6, col = cols[1], lwd = 2, ylim=c(0,6),
      xlab="x", ylab="f(x)",
      main=expression(paste("Beta pdf, ", alpha, " < ", beta)))
for (i in 2:6) curve(dbeta(x,ab[[i]][1],ab[[i]][2]),add = TRUE, col = cols[i], lwd = 2)
legend("top", legend = sapply(ab, function(p)paste0("a=",p[1]," b=", p[2])), col = cols, lwd = 2)


set.seed(42)
n <- 1000
lam <- 3
ind <- rbinom(n, size=1e6, prob = lam/1e6)
clu <- replicate(n, {k <- rpois(1,1); sum(1+rpois(k, 2))})

dis = function(y) var(y)/mean(y)
c("Var/Mean" ,indipendent = dis(ind), clustered = dis(clu))

show <- function(y, main){
  m <- max(y)
  hist(y, breaks = seq(-0.5, m + 0.5, 1),freq = FALSE, col = "lightblue", main = main, xlab = "weekly admissions")
  points(0:m, dpois(0:m, mean(y)), type="b", col = "red", pch = 16, lwd = 2)
  legend("topright", c("observed", "Poisson(mean)"),
         fill = c("lightblue", NA), border = c("black", NA),
         col = c(NA, "red"), lty = c(NA, 1), pch = c(NA, 16), cex = 0.8)
}
par(mfrow=c(2,2))
show(ind, "Independent case: Poisson fits")
plot(ind[1:52], type = "h", xlab= "week", ylab = "admissions", main = "Indipendent")
show(clu, "Clusetered case: Poisson fails")
plot(clu[1:52], type = "h", xlab= "week", ylab = "admissions", main = "Dipendent")
par(mfrow=c(1,1))

substance = read.table("http://stat4ds.rwth-aachen.de/data/Substance.dat", header=TRUE)
# Collapse over cigarettes
tab <- tapply(substance$count, list(substance$alcohol, substance$marijuana), sum)
tab

success <- tab[c("yes", "no"), "yes"]     # marijuana users in each alcohol group
total   <- rowSums(tab)[c("yes", "no")]

p_hat <- success / total
d     <- p_hat[1] - p_hat[2]
se    <- sqrt(sum(p_hat * (1 - p_hat) / total))
d + c(-1, 1) * qnorm(0.975) * se

# Built-in (Wald-type for a difference when correct = FALSE)
prop.test(success, total, correct = FALSE)$conf.int


cols <- c("red", "blue", "green", "yellow", "orange")
lo <- 1e-6; hi <- 1 - 1e-6

plot_beta <- function(ab, main, ylim = c(0, 6)) {
  curve(dbeta(x, ab[[1]][1], ab[[1]][2]), lo, hi, col = cols[1], lwd = 2,
        ylim = ylim, xlab = "x", ylab = "f(x)", main = main)
  for (i in 2:length(ab))
    curve(dbeta(x, ab[[i]][1], ab[[i]][2]), add = TRUE, col = cols[i], lwd = 2)
  legend("top", legend = sapply(ab, function(p) paste0("a=", p[1], " b=", p[2])),
         col = cols[seq_along(ab)], lwd = 2, cex = 0.7)
}

# alpha > beta: hump x3, beta = 1, alpha = 1, J-shaped, U-shaped
ab_gt <- list(c(20,5), c(5,2), c(2,1), c(2,0.5), c(0.7,0.2))
# alpha < beta: the mirror images
ab_lt <- list(c(5,20), c(2,5), c(1,2), c(0.5,2), c(0.2,0.7))

par(mfrow = c(1, 2))
plot_beta(ab_gt, expression(paste("Beta pdf, ", alpha, " > ", beta)))
plot_beta(ab_lt, expression(paste("Beta pdf, ", alpha, " < ", beta)))
par(mfrow = c(1, 1))

y <- seq(0, 1, length.out = 5)
dbeta(y, 1, 1)
dunif(y, 0, 1)
par(mfrow=c(1,1))
curve(dbeta(x, 1, 1), 0, 1, col = "blue", lwd = 4, ylim = c(0, 2),
      xlab = "y", ylab = "f(y)", main = "Beta(1,1) vs Unif(0,1)")
curve(dunif(x, 0, 1), add = TRUE, col = "red", lwd = 2, lty = 2)
legend("topright", c("Beta(1,1)", "Unif(0,1)"), col = c("blue", "red"),
       lwd = c(4, 2), lty = c(1, 2))

n <- 1e5
par(mfrow = c(1, 2))
hist(rbeta(n, 1, 1), freq = FALSE, breaks = 30, col = "blue",
     ylim = c(0, 2), xlab = "y", main = "Beta(1,1) sample")
abline(h = 1, col = "red", lwd = 2)
hist(runif(n, 0, 1), freq = FALSE, breaks = 30, col = "blue",
     ylim = c(0, 2), xlab = "y", main = "Unif(0,1) sample")
abline(h = 1, col = "red", lwd = 2)
par(mfrow = c(1, 1))

y <- seq(0, 1, length.out = 200)

par(mfrow = c(1, 2))
plot(y, dbeta(y, 1, 1), type = "l", col = "blue", lwd = 2, ylim = c(0, 2),
     xlab = "y", ylab = "f(y)", main = "Beta(1,1)")
plot(y, dunif(y, 0, 1), type = "l", col = "blue", lwd = 2, ylim = c(0, 2),
     xlab = "y", ylab = "f(y)", main = "Unif(0,1)")
par(mfrow = c(1, 1))
