getwd()
setwd("./Repository/R")

load("./data-20240319/feeling.Rdata")
if (!exists("ggplot2")) install.packages("ggplot2")
if (!exists("dplyr")) install.packages("dplyr")
if (!exists("tidyr")) install.packages("tidyr")
if (!exists("httpgd")) install.packages("httpgd")
library(ggplot2)
library(dplyr)
library(tidyr)
library(httpgd)

feeling <- drop_na(feeling, ft_immig_2016, ft_white_2016, ft_black_2016)

score <- feeling$ft_immig_2016
white_feature <- feeling$ft_white_2016
black_feature <- feeling$ft_black_2016



asfeeling <- cut(score, 4,
                 c("Strongly unfavorable","Unfavorable","Lightly Favorable",
                   "Strongly Favorable"))

ft_immig_2016_v2 <- data.frame(score, asfeeling)

# plot solid line, set plot size, but omit axes
plot(x = dnorm(black_feature, mean = mean(black_feature)),
     y = white_feature, type = "l", lty = 1, ylim = c(0, 100),
     axes = FALSE, bty = "n", xaxs = "i", yaxs = "i", main = "Ratio",
     xlab = "ft_black_2016", ylab = "ft_white_2016")

# plot dashed line
lines(x=seq(black_feature), y=seq(white_feature), lty=2)

# add axes
axis(side=1, labels=black_feature, at=seq(black_feature))
axis(side=2, at=seq(5,101,5), las=1)

# add legend
par(xpd=TRUE)
legend(x=1.5, y=2, legend=c("solid", "dashed"), lty=1:2, box.lty=0, ncol=2)


ft_immig_2016_v2 <- na.omit(ft_immig_2016_v2)

write.csv(ft_immig_2016_v2, file = "./data-20240319/ft_immig_2016_v2.csv")

save(ft_immig_2016_v2,file="./data-20240319/ft_immig_2016_v2.RData")
