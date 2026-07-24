getwd()
setwd("C:/Users/mutua/Documents/Repository/R")

load("./data-20240319/feeling.Rdata")

score <- feeling$ft_immig_2016

asfeeling <- cut(score, 4,
                 c("Strongly unfavorable","Unfavorable","Lightly Favorable",
                   "Strongly Favorable"))

ft_immig_2016_v2 <- data.frame(score, asfeeling)

# plot solid line, set plot size, but omit axes
plot(x=dnorm(feeling$ft_black_2016,mean = mean(feeling$ft_black_2016), y=feeling$ft_white_2016, type="l", lty=1, ylim=c(0,100),
     axes=F, bty="n", xaxs="i", yaxs="i", main="Ratio",
     xlab="ft_black_2016", ylab="ft_white_2016")

# plot dashed line
lines(x=seq(feeling$ft_black_2016), y=seq(feeling$ft_white_2016), lty=2)

# add axes
axis(side=1, labels=feeling$ft_black_2016, at=seq(feeling$ft_black_2016))
axis(side=2, at=seq(5,101,5), las=1)

# add legend
par(xpd=TRUE)
legend(x=1.5, y=2, legend=c("solid", "dashed"), lty=1:2, box.lty=0, ncol=2)


ft_immig_2016_v2 <- na.omit(ft_immig_2016_v2)

write.csv(ft_immig_2016_v2, file = "./data-20240319/ft_immig_2016_v2.csv")

save(ft_immig_2016_v2,file="./data-20240319/ft_immig_2016_v2.RData")
