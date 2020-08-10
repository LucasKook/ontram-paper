# ontram CI vs MCC learning efficicency
# 03.07.2020
# Lucas Kook


# Dependencies ------------------------------------------------------------

library(ontram)
library(tidyverse)
source("wine/get_data_uci.R")

# Data --------------------------------------------------------------------

dat <- get_data_wine("wine/", x_scale = TRUE)
dat$y_trainc <- preproc(dat$y_train)
dat$y_testc <- preproc(dat$y_test)

# Subsample data ----------------------------------------------------------

n <- nrow(dat$X_train)
nrs <- 10
idx <- sample(rep(seq_len(nrs), ceiling(n/nrs)), n)
tidx <- which(idx == 1)
dat$x_sub <- dat$X_train[tidx, ]
dat$y_sub <- dat$y_trainc[tidx, ]

# Params ------------------------------------------------------------------

ep <- 200
nb <- ceiling(n/nrs/16)
tlr <- 7e-4

# Funs --------------------------------------------------------------------

common <- function() {
  keras_model_sequential() %>%
    layer_dense(units = 16, activation = "relu", input_shape = ncol(dat$X_train)) %>%
    layer_dropout(0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu")
}

train_models <- function(nrs = 10, dd = dat, epochs = 100, batch_size = 16) {
  n <- nrow(dd$X_train)
  idx <- sample(rep(seq_len(nrs), ceiling(n/nrs)), n)
  nb <- ceiling(n/nrs/batch_size)
  ep <- epochs
  neff <- n/nrs
  ret <- list()
  for (fold in 1:nrs) {
    tidx <- which(idx == fold)
    dd$x_sub <- dd$X_train[tidx, ]
    dd$y_sub <- dd$y_trainc[tidx, ]
    mbl <- common() %>%
      layer_dense(units = ncol(dd$y_trainc) - 1L)

    mod <- ontram(mod_bl = mbl, mod_sh = NULL, mod_im = NULL, y_dim = ncol(dd$y_trainc),
                  x_dim = NULL, img_dim = NULL, method = "logit", n_batches = nb,
                  epochs = ep, response_varying = TRUE, lr = tlr)

    hist <- fit_ontram2(model = mod, history = TRUE, x_train = NULL, img_train = dd$x_sub,
                        y_train = dd$y_sub, x_test = NULL, img_test = dd$X_test,
                        y_test = dd$y_testc)

    mod3 <- common() %>%
      layer_dense(units = ncol(dd$y_trainc), activation = "softmax")
    compile(mod3, optimizer = optimizer_adam(lr = tlr),
            loss = "categorical_crossentropy")
    hist3 <- fit(mod3, x = dd$x_sub, y = dd$y_sub, epochs = ep,
                 batch_size = ceiling(nrow(dd$X_train)/nb),
                 validation_data = list(dd$X_test, dd$y_testc))
    ret[[fold]] <- data.frame(
      epoch = 1:ep,
      rv_train_loss = hist$train_loss,
      rv_test_loss = hist$test_loss,
      bl_train_loss = hist3$metrics$loss,
      bl_test_loss = hist3$metrics$val_loss
    )
  }
  return(ret)
}

# Run ---------------------------------------------------------------------

cases <- c(28, 14, 7, 3, 1)

tmp <- sapply(cases, function(x) train_models(nrs = x, epochs = ep))

out <- tmp %>%
  map_df(~as.data.frame(map_df(.x, ~as.data.frame(.x), .id = "fold")), .id = "case") %>%
  gather("model", "loss", rv_train_loss:bl_test_loss) %>%
  separate(model, c("name", "mode", "tmp"), remove = FALSE) %>%
  select(-tmp) %>%
  mutate(cases = factor(case, levels = 1:length(cases),
                        labels = paste0("n==", ceiling(nrow(dat$X_train)/cases))))

summ <- out %>%
  group_by(case, epoch, name, mode) %>%
  summarise(median_loss = median(loss)) %>%
  mutate(cases = factor(case, levels = 1:length(cases),
                       labels = paste0("n==", ceiling(nrow(dat$X_train)/cases))))
