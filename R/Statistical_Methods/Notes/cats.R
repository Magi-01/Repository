penguins = read.csv("penguins.csv")
plot(penguins$bill_length_mm,penguins$body_mass_g, col = factor(penguins$species))
fit = lm(body_mass_g ~ bill_length_mm + species, penguins)
summary(fit)

par(mfrow = c(2,2))
plot(fit)

par(mfrow = c(1,1))
plot(penguins$bill_length_mm,penguins$body_mass_g, col = factor(penguins$species))+
  abline(a=fit$coefficients["(Intercept)"], b=fit$coefficients["bill_length_mm"])+
  abline(a=fit$coefficients["(Intercept)"]+fit$coefficients["speciesGentoo"], b=fit$coefficients["bill_length_mm"], col=factor(penguins$species))

library(GGally)
ggpairs(penguins)

install.packages(ggpairs)
