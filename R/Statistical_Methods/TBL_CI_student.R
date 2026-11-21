## =====================================================================
## Group activity - Confidence intervals
## =====================================================================
## HOW TO WORK
## - Lines with TODO are yours to complete.
## - Each PART ends with questions: write the answers in comments.
##   You will report them in the final discussion.
## =====================================================================


## =====================================================================
## PART 1 - What drives the width of a CI for the mean?
## Y_i ~ N(mu, sigma^2) with sigma KNOWN. The CI for mu is
##     ybar +/- z_{1-alpha/2} * sigma / sqrt(n)
## Let's see how its width (= upper - lower bound) changes with
## the sample size n, the variability sigma and the confidence level.
## =====================================================================

ci_width <- function(n = 25, sigma = 1, conf_level = 0.95) {
  alpha  <- 1 - conf_level
  z_crit <- qnorm(1-alpha/2)                
  width  <- 2 * z_crit * sigma / sqrt(n)  # upper - lower bound
  return(width)
}

## 1a. Sample size (sigma = 1 and conf. level = 95% fixed)
n_vals   <- c(10, 25, 100, 500)
widths_n <- sapply(n_vals, function(n) ci_width(n = n))
names(widths_n) <- paste0("n=", n_vals)
widths_n

## 1b. Confidence level (n = 25 and sigma = 1 fixed)
cl_vals   <- c(.90, .95, .98, .99)
widths_cl <- sapply(cl_vals, function(n) ci_width(conf_level = n))                # same idea as 1a, now varying conf_level
names(widths_cl) <- paste0("cl=", cl_vals)
widths_cl

## 1c. Quick check: ratio of the widths for n = 25 and n = 100
0.7839856/0.3919928


## QUESTIONS (answer in comments)
## Q1. When n quadruples (25 -> 100), does the width halve, or change by
##     another factor? Why? 
Yeah it halves cause of the 4* factor in the sqrt(n)
## Q2. What happens to the width when the confidence level goes from 95% to 99%
##     for the same data? What do we gain and what do we pay?
## Q3. A colleague says: "to get a CI 10 times narrower I just need
##     10 times more data". Is she right? How many times more data are needed?
No because the factor is quadratic from sqrt(n)


## =====================================================================
## PART 2 - CI for a proportion
## For a proportion pi, the (Wald) CI based on the normal approximation is
##     pi.hat +/- z_{1-alpha/2} * sqrt(pi.hat * (1 - pi.hat) / n)
## The margin of error is ME = z * sqrt(pi.hat * (1 - pi.hat) / n).
## =====================================================================

## A newspaper reports a poll on the share of voters supporting a law:
## "the 95% confidence interval for pi is (0.38, 0.46)". Nothing else
## is said: no pi.hat, no n.
lower <- 0.38
upper <- 0.46
z     <- qnorm(0.975)

## 2a. Recover the point estimate and the margin of error from the endpoints.
pi_hat <- (upper+lower)/2
ME     <- (upper-lower)/2
c(pi_hat = pi_hat, ME = ME)

## 2b. Recover the sample size n of the poll
##     (hint: solve the expression of ME for n).
n_poll <- ceiling((4*pi_hat*(1-pi_hat))/ME**2)
n_poll

## 2c. The newspaper wants a follow-up poll with ME = 0.02 at 95% confidence.
##     How many respondents are needed
##     (i)  using the pi_hat you found in 2a?
##     (ii) with no prior information?
calc_n_prop <- function(ME, conf_level = 0.95, p_approx = 0.5) {
  alpha  <- 1 - conf_level
  z_crit <- qnorm(1 - alpha / 2)
  ceiling(z_crit^2 * p_approx * (1 - p_approx) / ME^2)
}
n_plugin <- calc_n_prop(0.2,p_approx = pi_hat)
n_noprior  <- calc_n_prop(0.2, p_approx = 0.5) #worst case if you want to be confident if not sure about data quantity
c(plugin = n_plugin, worst_case = n_noprior)


## 2d. CHALLENGE. A second poll, on n = 500 people, reports the interval
##     (0.40, 0.46) but forgets to say the confidence level.
##     Find pi_hat, the implied critical value z and the confidence level.
lower2 <- 0.40; upper2 <- 0.46; n2 <- 500
pi_hat2 <- TODO
se2     <- TODO
z2      <- TODO
conf2   <- TODO
c(pi_hat = pi_hat2, z = z2, conf_level = conf2)

## QUESTIONS
## Q4. Halving the margin of error (e.g. 0.04 -> 0.02): does the required n
##     halve, quadruple, or what? Check it with calc_n_prop().
## Q5. Why is p_approx = 0.5 the "safe" choice when we know nothing about pi?
##     Compare n_plugin and n_worst: which one is larger, and why?


## =====================================================================
## PART 3 - CI for a difference of means 
## Two independent groups, large-sample (normal) approximation.
## =====================================================================

set.seed(2026)
group_A <- rnorm(30, mean = 15, sd = 4)
group_B <- rnorm(35, mean = 12, sd = 5)

mean_a <- mean(group_A); s2_a <- var(group_A); n_a <- length(group_A)
mean_b <- mean(group_B); s2_b <- var(group_B); n_b <- length(group_B)

delta_hat <- mean_a - mean_b                     
se_delta  <- sqrt(s2_a / n_a + s2_b / n_b)       

## 3a. 95% CI for mu_A - mu_B
z_975    <- qnorm(0.975)
ci_lower <- delta_hat - z_975*se_delta
ci_upper <- delta_hat + z_975*se_delta
c(Lower = ci_lower, Upper = ci_upper)

## 3b. Repeat at 99%. Does the conclusion about zero change?
z_99    <- qnorm(0.99)
ci_lower <- delta_hat - z_99*se_delta
ci_upper <- delta_hat + z_99*se_delta
c(Lower = ci_lower, Upper = ci_upper)

## QUESTIONS
## Q7. Does the 95% CI contain 0? What can we say about the difference
##     between the two groups?
## Q8. The true difference is mu_A - mu_B = 3. Does your interval contain it?
##     If we repeated the whole procedure 1000 times, in how many cases
##     would we expect the interval to contain 3? Be careful with the
##     interpretation: what is random, the interval or mu?




## =====================================================================
## FINAL QUESTIONS
## 1. Which three ingredients control the width of a CI? Which one can we
##    actually choose when planning a study?
## 2. From a reported CI alone, what can and cannot be recovered
##    (pi_hat, ME, n, confidence level)?
## 3. What does "95% confidence" mean, and what does it NOT mean?
## =====================================================================