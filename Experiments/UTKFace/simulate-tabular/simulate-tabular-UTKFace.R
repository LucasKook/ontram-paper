# Simulation utilities for UTKFace data
# 20.05.2020
# Lucas Kook


# Dependencies ------------------------------------------------------------

library(tram)

# Data --------------------------------------------------------------------

bpath <- "data/"
bfile <- "UTKFace"
opath <- paste0(bpath, "simulated_from_", bfile, ".csv")
response <- "age_group"
dat <- read.csv(paste0(bpath, bfile, ".csv"))
dat[, response] <- ordered(dat[, response],
                           levels = c("[0, 4)", "[4, 13)", "[13, 20)",
                                      "[20, 31)", "[31, 46)", "[46, 61)",
                                      "[61, 117)"))

# Null model --------------------------------------------------------------

fm <- as.formula(paste0(response, "~ 1"))
m0 <- Polr(fm, data = dat)
cfb <- coef(m0, with_baseline = TRUE)

# Parameters --------------------------------------------------------------

beta <- log(c(1, 1.2, 1/1.2, 1, 1.5, 1/1.5, 1, 1, 1, 1))
p <- length(beta)
sds <- rep(1.55, p)
n <- nrow(dat)

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

m <- Polr(age_group ~ . - id - path, data = newdat[,-1L])
coef(m, with_baseline = TRUE)
