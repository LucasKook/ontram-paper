# QWK model for the wine data
# 03.07.2020
# Lucas Kook


# Dependencies ------------------------------------------------------------

library(ontram)
source("wine/get_data_uci.R")

# Loop over folds ---------------------------------------------------------

ep <- 1000
nb <- 1
ce_train <- c()
ce_test <- c()
confmats_train <- array(0, dim = c(6, 6, 20))
confmats_test <- array(0, dim = c(6, 6, 20))
qwk_train <- c()
qwk_test <- c()
qwk_metric <- get_metric(6, 2)
qwkloss <- get_loss(weight_scheme(6, 2))
rps_train <- c()
rps_test <- c()

for (split in 0:19) {
  # Data
  dat <- get_data_wine("wine/", x_scale = TRUE, split_num = split)
  dat$y_trainc <- preproc(dat$y_train)
  dat$y_testc <- preproc(dat$y_test)
  # Model
  mod <- keras_model_sequential() %>%
    layer_dense(units = 16, activation = "relu", input_shape = ncol(dat$X_train)) %>%
    layer_dropout(0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = ncol(dat$y_trainc), activation = "softmax")
  compile(mod, optimizer = "adam", loss = qwkloss,
          metrics = c("accuracy", "categorical_crossentropy", qwk_metric))
  fit(mod, x = dat$X_train, y = dat$y_trainc, batch_size = ceiling(nrow(dat$y_trainc)/nb),
      epochs = ep, view_metrics = FALSE)
  # Eval
  eval_train <- evaluate(mod, x = dat$X_train, y = dat$y_trainc)
  eval_test <- evaluate(mod, x = dat$X_test, y = dat$y_testc)
  preds_train <- predict(mod, x = dat$X_train, y = dat$y_trainc)
  preds_test <- predict(mod, x = dat$X_test, y = dat$y_testc)
  ce_train <- append(ce_train, eval_train[["categorical_crossentropy"]])
  ce_test <- append(ce_test, eval_test[["categorical_crossentropy"]])
  tgt1 <- factor(apply(dat$y_trainc, 1, which.max), levels = 1:6)
  prd1 <- factor(apply(preds_train, 1, which.max), levels = 1:6)
  confmats_train[,, split + 1] <- get_confmat(tgt1, prd1)
  tgt <- factor(apply(dat$y_testc, 1, which.max), levels = 1:6)
  prd <- factor(apply(preds_test, 1, which.max), levels = 1:6)
  confmats_test[,, split + 1] <- get_confmat(tgt, prd)
  qwk_train <- append(qwk_train, eval_train[["qwk"]])
  qwk_test <- append(qwk_test, eval_test[["qwk"]])
  rps_train <- append(rps_train, rps(dat$y_trainc, preds_train))
  rps_test <- append(rps_test, rps(dat$y_testc, preds_test))
}

dqwk_train <- apply(confmats_train, 3, compute_kappa, weights = weight_scheme(6, 2))
dqwk_test <- apply(confmats_test, 3, compute_kappa, weights = weight_scheme(6, 2))

acc_train <- apply(confmats_train, 3, function(x) sum(diag(x)) / sum(x))
acc_test <- apply(confmats_test, 3, function(x) sum(diag(x)) / sum(x))

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
