df = read.csv("sandwich.csv", header = T)

# -------------------------------
# Load libraries
# -------------------------------
library(ggplot2)
library(dplyr)
library(boot)

# Check structure
str(df)
summary(df)

# Convert predictors to factors
df$bread   <- factor(df$bread)
df$topping <- factor(df$topping)
df$butter  <- factor(df$butter)

str(df)
summary(df)
# Quick check for missing values
sum(is.na(df))

# -------------------------------
# Visualizations
# -------------------------------
par(mfrow = c(2,2))
# Distribution of ant counts
ggplot(df, aes(x = antCount)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  theme_minimal()

# Boxplots by bread / topping / butter
ggplot(df, aes(x = bread, y = antCount, fill = bread)) +
  geom_boxplot() + theme_minimal()

ggplot(df, aes(x = topping, y = antCount, fill = topping)) +
  geom_boxplot() + theme_minimal()

ggplot(df, aes(x = butter, y = antCount, fill = butter)) +
  geom_boxplot() + theme_minimal()

# -------------------------------
# Linear Regression Model
# -------------------------------
# Linear model
model_lm <- lm(antCount ~ bread + topping + butter, data = df)
summary(model_lm)

# Diagnostic plots
par(mfrow = c(2,2))
plot(model_lm)

library(emmeans)
emm <- emmeans(model_lm, ~ bread + topping + butter)

plot(emm)
