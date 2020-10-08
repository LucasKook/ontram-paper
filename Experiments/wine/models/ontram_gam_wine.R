# Emulate GAMs in ontram via keras API
# Lucas Kook
# 17.09.2020


# Dependencies ------------------------------------------------------------

library(ontram)
source("wine/get_data_uci.R")

# Parameters --------------------------------------------------------------

ep <- 1200
nb <- 8
tlr <- 1e-3
npred <- 11
l2 <- 0

# Data --------------------------------------------------------------------

dat <- get_data_wine("wine/", x_scale = TRUE, split_num = 0L)
dat$y_trainc <- preproc(dat$y_train)
dat$y_testc <- preproc(dat$y_test)

# Model -------------------------------------------------------------------

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

theta_to_gamma <- function(gammas) {
  thetas <- gammas
  thetas[2:length(thetas)] <- log(diff(gammas))
  thetas[is.nan(thetas)] <- - 1e20
  return(thetas)
}

# Format input data -------------------------------------------------------

xtr <- list()
for (idx in 1:npred) {
  xtr[[idx]] <- dat$X_train[, idx, drop = FALSE]
}

xte <- list()
for (idx in 1:npred) {
  xte[[idx]] <- dat$X_test[, idx, drop = FALSE]
}

# Ensemble ----------------------------------------------------------------

nruns <- 20
nlls <- c()
fits <- list()
cfb <- list()

for (run in 1:nruns) {
  ms <- list()
  for (idx in 1:npred) {
    ms[[idx]] <- common1()
  }
  
  mbl <- mod_baseline(ncol(dat$y_trainc))
  # get_weights(mbl)
  # tmp <- get_weights(mbl)
  # tmp[[1]][] <- theta_to_gamma(c(-1, 0.98, 4.73, 7.69, 10.87))
  # set_weights(mbl, tmp)
  # freeze_weights(mbl)
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
  preds <- predict(mod, x = NULL, y = dat$y_testc, im = xte)
  matplot(t(preds$pdf), type = "l")
  nlls[run] <- preds$negLogLik
  xx <- seq(0, 1, length.out = 1e3)
  out <- lapply(ms, function(x) x(matrix(xx)))
  fits[[run]] <- do.call("cbind", lapply(out, function(x) x$numpy()))
  cfb[[run]] <- get_weights(mod$mod_baseline)[[1]]
  pout <- paste0("wine/results/gam/fits_run", run, ".csv")
  write.csv(fits[[run]], pout, quote = FALSE, row.names = FALSE)
  pout2 <- paste0("wine/results/gam/cfb_run", run, ".csv")
  write.csv(cfb[[run]], pout2, quote = FALSE, row.names = FALSE)
}

write.csv(nlls, "wine/results/gam/nlls.csv", quote = FALSE, row.names = FALSE)

