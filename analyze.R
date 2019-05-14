bottom_up <- read.csv("bottom-up.csv", header=TRUE)
pointer_generator <- read.csv("pointer-generator.csv", header=TRUE)
lin_reg <- function(dat) {
  fit <- lm(score ~ average_copy_length, data=dat)
  print(summary(fit))
  plot(dat$average_copy_length, dat$score)
  abline(fit)
}
stats <- function(dat) {
  print(mean(dat$average_copy_length))
  print(mean(dat$score))
}
lin_reg(bottom_up)
lin_reg(pointer_generator)
print("bottom_up")
stats(bottom_up)
print("pointer_generator")
stats(pointer_generator)
