# Quick illustration of impropriety of the QWK loss
# 05.07.2020
# Lucas Kook

weight_scheme <- function(K, p) {
  outer(1:K, 1:K, function(x, y) abs(x - y)^p / (K - 1)^p)
}

kappa_loss <- function(target, predicted, weights) {
  idx <- apply(target, 1, which.max)
  wgts <- weights[idx, , drop = FALSE]
  num <- sum(wgts * predicted)

  fi <- colSums(target) / sum(target)
  den <- sum(fi * weights %*% colSums(predicted))

  ret <- log(num) - log(den)
  return(ret)
}

expected_score <- function(target, predicted, p = 2) {
  score <- kappa_loss(target, predicted, weight_scheme(ncol(target), p))
  ret <- sum(score * target)
  return(ret)
}

# Simple example (same as in main text)
tden <- matrix(c(0.3, 0.4, 0.3), nrow = 1)
pden <- matrix(c(0.1, 0.8, 0.1), nrow = 1)
expected_score(tden, pden) <= expected_score(tden, tden)
1 - exp(expected_score(tden, pden)) >= 1 - exp(expected_score(tden, tden))
