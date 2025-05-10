
df = read.csv("sandwich.csv", header = T)

#library(ggplot2)

ggplot(df, aes(x = antCount, y = bread)) +
  geom_line() +
  labs(title = "Line Plot", x = "X-axis", y = "Y-axis") +
  theme_minimal()

ggplot(df, aes(x = bread, y = antCount)) +
  geom_col() +
  labs(title = "Value Bar Plot", x = "bread", y = "antCount") +
  theme_minimal()

as.

df["antCount"] = as.numeric(as.integer(unlist(df["antCount"])))
df["bread"] = as.numeric(as.factor(unlist(df["bread"])))
df["topping"] = as.numeric(as.factor(unlist(df["topping"])))
df["butter"] = as.numeric(as.factor(unlist(df["butter"])))

library(nnet)
model <- multinom(unlist(df["bread"]) ~ unlist(df["antCount"]) * unlist(df["topping"]) * unlist(df["butter"]), data = df)

df$predicted <- predict(model, newdata = df)

ggplot(df, aes(x = antCount, y = bread, color = factor(topping))) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ butter) +
  labs(title = "3-Way Interaction: x * z * k", color = "z")
