# ontram ci model for the wine data under permuted class labels
# 03.07.2020
# Lucas Kook


# Dependencies ------------------------------------------------------------

library(tidyverse)
library(ontram)
source("wine/get_data_uci.R")

# Data --------------------------------------------------------------------

dat <- get_data_wine("wine/", x_scale = TRUE)
dat$y_trainc <- preproc(dat$y_train)
dat$y_testc <- preproc(dat$y_test)

# Permute classes ---------------------------------------------------------

idx <- c(6, 2, 5, 1, 4, 3)
dat$ytrp <- dat$y_trainc[, idx]
dat$ytep <- dat$y_testc[, idx]
ep <- 150
nb <- 125
tlr <- 5e-4

# Model -------------------------------------------------------------------

common <- function() {
  keras_model_sequential() %>%
    layer_dense(units = 16, activation = "relu", input_shape = ncol(dat$X_train)) %>%
    layer_dropout(0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(0.3) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 16, activation = "relu")
}

out <- list()

for (run in 1:20) {
  mbl <- common() %>%
    layer_dense(units = ncol(dat$y_trainc) - 1L)

  mod <- ontram(mod_bl = mbl, mod_sh = NULL, mod_im = NULL, y_dim = ncol(dat$y_trainc),
                x_dim = NULL, img_dim = NULL, method = "logit", n_batches = nb,
                epochs = ep, response_varying = TRUE, lr = tlr)

  hist <- fit_ontram2(model = mod, history = TRUE, x_train = NULL, img_train = dat$X_train,
                      y_train = dat$y_trainc, x_test = NULL, img_test = dat$X_test,
                      y_test = dat$y_testc)

  mbl2 <- common() %>%
    layer_dense(units = ncol(dat$y_trainc) - 1L)

  mod2 <- ontram(mod_bl = mbl2, mod_sh = NULL, mod_im = NULL, y_dim = ncol(dat$y_trainc),
                 x_dim = NULL, img_dim = NULL, method = "logit", n_batches = nb,
                 epochs = ep, response_varying = TRUE, lr = tlr)

  hist2 <- fit_ontram2(model = mod2, history = TRUE, x_train = NULL, img_train = dat$X_train,
                       y_train = dat$ytrp, x_test = NULL, img_test = dat$X_test,
                       y_test = dat$ytep)

  mod3 <- common() %>%
    layer_dense(units = ncol(dat$y_trainc), activation = "softmax")
  compile(mod3, optimizer = optimizer_adam(lr = tlr),
          loss = "categorical_crossentropy")
  hist3 <- fit(mod3, x = dat$X_train, y = dat$y_trainc, epochs = ep,
               batch_size = ceiling(nrow(dat$X_train)/nb),
               validation_data = list(dat$X_test, dat$y_testc))

  mod4 <- common() %>%
    layer_dense(units = ncol(dat$y_trainc), activation = "softmax")
  compile(mod4, optimizer = optimizer_adam(lr = tlr),
          loss = "categorical_crossentropy")
  hist4 <- fit(mod4, x = dat$X_train, y = dat$ytrp, epochs = ep,
               batch_size = ceiling(nrow(dat$X_train)/nb),
               validation_data = list(dat$X_test, dat$ytep))

  # Save histories ----------------------------------------------------------

  ret <- data.frame(
    epoch = 1:ep,
    model = rep(c("rv", "softmax"), each = 2 * ep),
    permuted = factor(rep(c("true", "permuted", "true", "permuted"), each = ep),
                      levels = c("true", "permuted")),
    set = rep(c("train", "test"), each = 4 * ep),
    loss = c(hist$train_loss, hist2$train_loss, hist3$metrics$loss, hist4$metrics$loss,
             hist$test_loss, hist2$test_loss, hist3$metrics$val_loss, hist4$metrics$val_loss)
  )
  out[[run]] <- ret
}

endout <- out %>%
  map_df(.x, .f = ~as.data.frame(.x), .id = "run")
