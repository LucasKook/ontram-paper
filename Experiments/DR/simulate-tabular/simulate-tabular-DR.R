# Simulation utilities for Diabetic Retinopathy data
# 20.05.2020
# Lucas Kook


# Dependencies ------------------------------------------------------------

library(tram)

# Data --------------------------------------------------------------------

dat <- read.csv("data/trainLabels.csv") # change path to trainLabels.csv
dat$level <- ordered(dat$level, levels = 0:4)

# Null model --------------------------------------------------------------

m0 <- Polr(level ~ 1, data = dat)
cfb <- coef(m0, with_baseline = TRUE)

# Parameters --------------------------------------------------------------

beta <- log(c(1, 1.2, 1/1.2, 1, 1.5, 1/1.5, 1, 1, 1, 1))
p <- length(beta)
sds <- rep(1.4, p)
n <- nrow(dat)
response <- "level"

# Functions ---------------------------------------------------------------

sim_covariate <- function(beta, response, data, sd) {
  fm <- as.formula(paste0("~ 0 + ", response))
  contr <- list("contr.treatment")
  names(contr) <- response
  X <- model.matrix(fm, data = data, contrasts.arg = contr)
  betas <- cumsum(rep(beta, ncol(X)))
  Y <- X %*% betas + rnorm(nrow(data), sd = sd)
  ret <- c(scale(Y, center = TRUE, scale = FALSE))
  return(ret)
}

sim_all <- function(betas, response, data, sds) {
  grid <- cbind(betas, sds)
  ret <- apply(grid, 1, function(parms, response, data) {
    beta <- parms[[1]]
    sd <- parms[[2]]
    tmp <- sim_covariate(beta = beta, response = response, data = data, sd = sd)
    return(tmp)
  }, response = response, data = data)
  return(ret)
}

# Final dataset -----------------------------------------------------------

set.seed(24101968)
nd <- sim_all(betas = beta, response = response, data = dat, sds = sds)
newdat <- cbind(dat, x = nd)
newdat <- janitor::clean_names(newdat)

m <- Polr(level ~ ., data = newdat[,-1L])
coef(m, with_baseline = TRUE)
