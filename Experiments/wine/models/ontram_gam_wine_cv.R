# ontram gam wine cross validation
# Lucas Kook
# 24.09.2020


# Dependencies ------------------------------------------------------------

library(ontram)
source("wine/get_data_uci.R")

# Loop over folds ---------------------------------------------------------

nm <- "ontram_gam_wine"
ep <- 1200
nb <- 8
tlr <- 1e-3
npred <- 11
l2 <- 0
ce_train <- c()
ce_test <- c()
confmats_train <- array(0, dim = c(6, 6, 20))
confmats_test <- array(0, dim = c(6, 6, 20))
qwk_train <- c()
qwk_test <- c()
rps_train <- c()
rps_test <- c()

# Architecture ------------------------------------------------------------

common1 <- function(dr = 0.2, l1 = 1, ll2 = 5) {
  keras_model_sequential() %>% 
    layer_dense(input_shape = 1L, units = 16L, activation = "relu", kernel_regularizer = regularizer_l1_l2(l1, ll2)) %>% 
    layer_dropout(dr) %>% 
    layer_dense(units = 16L, activation = "relu", kernel_regularizer = regularizer_l1_l2(l1, ll2)) %>% 
    layer_dropout(dr) %>% 
    layer_dense(units = 8L, activation = "relu", kernel_regularizer = regularizer_l1_l2(l1, ll2)) %>% 
    layer_dropout(dr) %>% 
    layer_dense(units = 1L, activation = "linear", kernel_regularizer = regularizer_l1_l2(l1, ll2))
}


for (split in 0:19) {
  # Data
  dat <- get_data_wine("wine/", x_scale = TRUE, split_num = split)
  dat$y_trainc <- preproc(dat$y_train)
  dat$y_testc <- preproc(dat$y_test)
  xtr <- list()
  for (idx in 1:npred) {
    xtr[[idx]] <- dat$X_train[, idx, drop = FALSE]
  }
  
  xte <- list()
  for (idx in 1:npred) {
    xte[[idx]] <- dat$X_test[, idx, drop = FALSE]
  }
  # Model
  ms <- list()
  for (idx in 1:npred) {
    ms[[idx]] <- common1()
  }
  mbl <- mod_baseline(ncol(dat$y_trainc))
  msh <- NULL
  mim <- keras_model(inputs = lapply(ms, function(x) x$input),
                     outputs = layer_concatenate(lapply(ms, function(x) x$output)) %>% 
                       layer_dense(units = 1L, use_bias = FALSE, trainable = FALSE, 
                                   kernel_initializer = initializer_constant(1)))
  mod <- ontram(mod_bl = mbl, mod_sh = msh, mod_im = mim, x_dim = ncol(dat$X_train),
                y_dim = ncol(dat$y_testc), method = "logit", n_batches = nb, 
                epochs = ep, lr = tlr)
  fit_ontram3(model = mod, history = FALSE, img_train = xtr,
              y_train = dat$y_trainc, img_test = xte, y_test = dat$y_testc,
              lambda2 = l2, numnet = npred)
  # Eval
  preds_train <- predict(mod, im = xtr, y = dat$y_trainc, x = NULL)
  preds_test <- predict(mod, im = xte, y = dat$y_testc, x = NULL)
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


write.csv(out, paste0("wine/results/", nm, "_metrics.csv"))
# kkmisc::write_results(out, extra = "metrics", path = "wine/results")
