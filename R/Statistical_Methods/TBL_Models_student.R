## =====================================================================
y    <- as.numeric((nhtemp - 32) / 1.8)   # degrees Celsius, as a plain vector
year <- 1912:1971
n    <- length(y)
plot(year, y, pch = 16, xlab = "Year", ylab = "y")

## Sample autocorrelation at lag 1 (the r1 of the slides)
r1 <- function(x) acf(x, lag.max = 1, plot = FALSE)$acf[2]
r1(y)                                      # observed value


## =====================================================================
## PART 1 - Sufficient statistics 
## For iid N(mu, sigma^2) data, t(y) = (mean(y), var(y)) is sufficient.
## Let's see what this means in practice.
## =====================================================================

## Three datasets: the real one, a SHUFFLED copy, and a FAKE one built to
## have exactly the same mean and variance as y but different values.
set.seed(1)
y_perm <- sample(y)
z      <- rnorm(n)
y_fake <- mean(y) + sd(y) * (z - mean(z)) / sd(z)

rbind(y = c(mean(y), var(y)), perm = c(mean(y_perm), var(y_perm)),
      fake = c(mean(y_fake), var(y_fake)))

## Log-likelihood = log of the joint density of the data, for given parameters
loglik_norm <- function(data, mu, sigma) sum(dnorm(data, mu, sigma, log = TRUE))
loglik_t5   <- function(data, mu, sigma) sum(dt((data - mu) / sigma, df = 5, log = TRUE) - log(sigma))

## A few candidate parameter values theta = (mu, sigma)
theta <- list(c(10, 1), c(10.7, 0.8), c(11, 1.5))

## 1a. Normal model: compare the three datasets
for (th in theta) {
  print(c(y    = loglik_norm(y,      th[1], th[2]),
          perm = loglik_norm(y_perm, th[1], th[2]),
          fake = loglik_norm(y_fake, th[1], th[2])))
}
## 1b. Same thing with the t5 model (heavier tails)
TODO   # copy the loop above, using loglik_t5

## 1c. Do the three datasets have the same r1?
c(r1(y), r1(y_perm), r1(y_fake))

## QUESTIONS 
## Q1. Under the normal model, what do y, y_perm and y_fake have in common
##     as far as information on (mu, sigma^2) is concerned? Link it to the
##     factorization f(y) = h(y) g_theta(t(y)) in the slides.
## Q2. Is (mean, var) still enough under the t5 model? Which dataset differs?
## Q3. y_perm and y have the same values in different order. What do ALL iid
##     models ignore? Is the order important in the real data?


## =====================================================================
## PART 2 - Simulate from Model 1 (normal) and Model 2 (t5) 
## =====================================================================
## Model 1: Y_i ~ N(mu, sigma^2)               estimates: mu = mean(y), sigma = sd(y)
## Model 2: (Y_i - mu)/sigma ~ t5              the variance of Y_i is sigma^2 * 5/3,

sim_m1 <- function() rnorm(n, mean = mean(y), sd = sd(y))
sim_m2 <- function() TODO     

## 2a. Compare one simulated dataset with the real data (as on the slides)
set.seed(2018)
plot(year, y, pch = 16, xlab = "Year", ylab = "y")
points(year, sim_m1(), col = 2, pch = 16)
## Which features of y does the red series fail to reproduce?

## 2b. Model checking with QQ plots
par(mfrow = c(1,2))
stats::qqnorm(y, main = "Model 1: normal"); stats::qqline(y)
stats::qqplot(qt(ppoints(n), df = 5), y,  xlab = "t5 quantiles",
              ylab = "y", main = "Model 2: t5")
stats::qqline(y, distribution = function(p) qt(p, df = 5))
par(mfrow = c(1, 1))

## 2c. A single dataset tells us little: simulate 2000 datasets from each
##     model and look at the distribution of r1.
set.seed(2026)
R    <- 2000
r_m1 <- replicate(R, r1(sim_m1()))
r_m2 <- replicate(R, r1(sim_m2()))
boxplot(list(M1 = r_m1, M2 = r_m2), ylab = "r1")
abline(h = r1(y), col = 2, lwd = 2)          # observed r1
## proportion of simulated datasets with r1 at least as large as observed:
c(M1 = mean(r_m1 >= r1(y)), M2 = mean(r_m2 >= r1(y)))

## 2d. YOUR OWN FEATURE: choose another feature of the data (e.g. max(y),
##     min(y), the number of years above 11.5, ...) and repeat 2c for it.
TODO

## QUESTIONS
## Q4. Which model describes the tails better? Which one reproduces r1?
## Q5. In which sense are mu and sigma "the same parameters" in Model 1 and 2?
##     (hint: var(Y_i) in each model)


## =====================================================================
## PART 3 - Relax "identically distributed": two periods 
## Model 3-lite: Y_i ~ N(mu_1, sigma^2) for 1912-1941,
##               Y_i ~ N(mu_2, sigma^2) for 1942-1971 (independent)
## =====================================================================
grp  <- rep(1:2, each = n / 2)
mu_g <- tapply(y, grp, mean)                 # one mean per period
fit  <- as.numeric(mu_g)[grp]                # mean assigned to each year
sigma3 <- sqrt(mean((y - fit)^2))            # spread around the period means

sim_m3 <- function() TODO                    # simulate from the assumed normal model

## 3a. Plot the data with the fitted period means and one simulated dataset
plot(year, y, pch = 16, xlab = "Year", ylab = "y")
lines(year, fit, col = 4, lwd = 2)
points(year, sim_m3(), col = 2, pch = 16)

## 3b. r1 under Model 3-lite
set.seed(2026)
r_m3 <- replicate(R, r1(sim_m3()))
boxplot(list(M1 = r_m1, M2 = r_m2, M3 = r_m3), ylab = "r1")
abline(h = r1(y), col = 2, lwd = 2)

## 3c. Model checking: QQ plot of the residuals y - fit
res <- y - fit
qqnorm(res, pch = 16, main = "Model 3-lite"); qqline(res)

## QUESTIONS
## Q6. Why does r1 go up although the simulated observations are independent?
## Q7. Is the model fully satisfactory? What is still missing?


## =====================================================================
## FINAL DISCUSSION 
##
## Model   | theta           | tails (QQ)   | r1 reproduced?   | verdict
## --------|-----------------|--------------|------------------|--------
## M1      | (mu, sigma^2)   |              |                  |
## M2      | (mu, sigma^2)   |              |                  |
## M3-lite | (mu1, mu2, s^2) |              |                  |
##
## 1. Which model would you keep, and why (Occam's razor)?
## 2. Which assumption does every model so far share, and fails to describe
##    the dependence between consecutive years?
## 3. What kind of model would you build next?
## =====================================================================
