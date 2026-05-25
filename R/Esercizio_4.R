library(MASS)
library(dplyr)
library(ggplot2)

str(cars)
attach(cars)

boxplot(speed~dist, data = cars)

tapply(speed, dist, mean)

dev_totale = sum((speed-mean(speed))^2)

dev_tot = dev_in + dev_between
b0 = mean(speed) - b1*mean(dist)
b1 = cov(dist,speed)/sd(dist)
y_hat = b0 + b1*x

dev_tot = y-mean(y)
dev_in = y-y_hat
dev_between = y_hat - mean(y)

deviance(lm(cars))

hist(speed, breaks=14, probability = T)
lines(density(speed, bw = h), col=2)
h = as.integer(0.9*min(IQR(speed),sd(speed))*length(speed)^(-1/5))
plot(speed~dist)
lines(lowess(dist,speed))
cor(cars)
cor = sum((speed-mean(speed))*(dist-mean(dist)))/((length(speed)-1)*sd(speed)*sd(dist))
cor = cov(speed,dist)/(sd(speed)*sd(dist))

fit1 = lm(speed~dist)
summary(fit1)

par(mfrow=c(1,2))
res = speed-b0-b1*speed
plot(fitted(fit1), res)
abline(h=0)
qqnorm(resid(fit1))
qqline(resid(fit1))

fit1log = lm(speed~log(dist))
fit1log$coefficients

predicted = predict(fit1log, newdata = cars,)

ggplot(mapping = aes(dist,speed), ) + geom_point() + geom_smooth(aes(colour = speed)) + 
  geom_line(aes(predicted))
