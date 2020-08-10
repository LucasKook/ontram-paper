# ontram CI model for the wine data
# 03.07.2020
# Lucas Kook


# Dependencies ------------------------------------------------------------

library(ontram)
source("wine/get_data_uci.R")

# Loop over folds ---------------------------------------------------------

nm <- "ontram_rv_wine"
ep <- 100
nb <- 250
ce_train <- c()
ce_test <- c()
confmats_train <- array(0, dim = c(6, 6, 20))
confmats_test <- array(0, dim = c(6, 6, 20))
qwk_train <- c()
qwk_test <- c()
rps_train <- c()
rps_test <- c()

for (split in 0:19) {
  # Data
  dat <- get_data_wine("wine/", x_scale = TRUE, split_num = split)
  dat$y_trainc <- preproc(dat$y_train)
  dat$y_testc <- preproc(dat$y_test)
  # Model
  mbl <- keras_model_sequential() %>%
    layer_dense(units = 16, activation = "relu", input_shape = ncol(dat$X_train)) %>%
    layer_dropout(0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = ncol(dat$y_trainc) - 1L)
  mod <- ontram(mod_bl = mbl, mod_sh = NULL, mod_im = NULL, y_dim = ncol(dat$y_trainc),
                x_dim = NULL, img_dim = NULL, method = "logit", n_batches = nb,
                epochs = ep, response_varying = TRUE)
  fit_ontram(model = mod, history = FALSE, x_train = NULL, img_train = dat$X_train,
             y_train = dat$y_trainc, x_test = NULL, img_test = dat$X_test,
             y_test = dat$y_testc)
  # Eval
  preds_train <- predict(mod, im = dat$X_train, y = dat$y_trainc, x = NULL)
  preds_test <- predict(mod, im = dat$X_test, y = dat$y_testc, x = NULL)
  ce_train <- append(ce_train, preds_train$negLogLik)
  ce_test <- append(ce_test, preds_test$negLogLik)
  tgt1 <- factor(apply(dat$y_trainc, 1, which.max), levels = 1:6)
  prd1 <- factor(preds_train$response, levels = 1:6)
  confmats_train[,, split + 1] <- get_confmat(tgt1, prd1)
  tgt <- factor(apply(dat$y_testc, 1, which.max), levels = 1:6)
  prd <- factor(preds_test$response, levels = 1:6)
  confmats_test[,, split + 1] <- get_confmat(tgt, prd)
  qwk_train <- append(qwk_train, 1 - exp(kappa_loss(dat$y_trainc, preds_train$pdf,
                                                    weight_scheme(6, 2))))
  qwk_test <- append(qwk_test, 1 - exp(kappa_loss(dat$y_testc, preds_test$pdf,
                                                  weight_scheme(6, 2))))
  rps_train <- append(rps_train, rps(dat$y_trainc, preds_train$pdf))
  rps_test <- append(rps_test, rps(dat$y_testc, preds_test$pdf))
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
