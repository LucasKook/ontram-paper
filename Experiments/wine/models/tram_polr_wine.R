# tram POLR model for the wine data
# 03.07.2020
# Lucas Kook


# Dependencies ------------------------------------------------------------

library(ontram)
library(tram)
source("wine/get_data_uci.R")

# Helpers -----------------------------------------------------------------

dat_to_df <- function(split_num = 0) {
  dat <- get_data_wine("wine/", x_scale = TRUE, split_num = split_num)
  dat$y_trainc <- preproc(dat$y_train, retfac = TRUE)
  dat$y_testc <- preproc(dat$y_test, retfac = TRUE)
  ydtr <- preproc(dat$y_train, retfac = FALSE)
  ydte <- preproc(dat$y_test, retfac = FALSE)
  dtrain <- data.frame(y = dat$y_trainc, x = dat$X_train)
  dtest <- data.frame(y = dat$y_testc, x = dat$X_test)
  return(list(train = dtrain, test = dtest,
              y_dummy_train = ydtr,
              y_dummy_test = ydte))
}

# Analysis ----------------------------------------------------------------

tmeth <- "logistic"
ce_train <- c()
ce_test <- c()
confmats_train <- array(0, dim = c(6, 6, 20))
confmats_test <- array(0, dim = c(6, 6, 20))
qwk_train <- c()
qwk_test <- c()
cfx <- matrix(0, nrow = 20, ncol = 16)
rps_train <- c()
rps_test <- c()

for (split in 0:19) {
  dat <- dat_to_df(split_num = split)
  m <- Polr(y ~ ., data = dat$train, method = tmeth)
  ce_train <- append(ce_train, - logLik(m)/nrow(dat$train))
  ce_test <- append(ce_test, - logLik(m, newdata = dat$test)/nrow(dat$test))
  preds_train <- predict(m, newdata = dat$train[, -1L], type = "density")
  preds_test <- predict(m, newdata = dat$test[, -1L], type = "density")
  cls_train <- ordered(apply(preds_train, 2, which.max), levels = 1:6)
  cls_test <- ordered(apply(preds_test, 2, which.max), levels = 1:6)
  confmats_train[,, split + 1L] <- get_confmat(dat$train$y, cls_train)
  confmats_test[,, split + 1L] <- get_confmat(dat$test$y, cls_test)
  qwk_train <- append(qwk_train, 1 - exp(kappa_loss(target = dat$y_dummy_train,
                                                    predicted = t(preds_train),
                                                    weights = weight_scheme(6, 2))))
  qwk_test <- append(qwk_test, 1 - exp(kappa_loss(target = dat$y_dummy_test,
                                                    predicted = t(preds_test),
                                                    weights = weight_scheme(6, 2))))
  rps_train <- append(rps_train, rps(dat$y_dummy_train, t(preds_train)))
  rps_test <- append(rps_test, rps(dat$y_dummy_test, t(preds_test)))
  cfx[split + 1, ] <- coef(m, with_baseline = TRUE)
}

acc_train <- apply(confmats_train, 3, function(x) sum(diag(x))/sum(x))
acc_test <- apply(confmats_test, 3, function(x) sum(diag(x))/sum(x))

dqwk_train <- apply(confmats_train, 3, compute_kappa, weights = weight_scheme(6, 2))
dqwk_test <- apply(confmats_test, 3, compute_kappa, weights = weight_scheme(6, 2))

out <- data.frame(
  split = 0:19,
  ce_train = ce_train,
  ce_test = ce_test,
  rps_train = rps_train,
  rps_test = rps_test,
  acc_train = acc_train,
  acc_test = acc_test,
  qwk_train = qwk_train,
  qwk_test = qwk_test,
  dqwk_train = dqwk_train,
  dqwk_test = dqwk_test
)
