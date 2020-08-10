# This script is adapted from https://github.com/tensorchiefs/dl_playr/tree/master/mlt
# Changes made compared to original script:
# - format to adhere to tidyverse style guide
# - add preproc() function
# - do not normalize outcome in load_data, because it's ordinal
# - delete all read function except for get_data_wine()
# 07.06.2020
# Lucas Kook

u_scale <- function (col, min_col, max_col, eps = 0.0) {
  return ((col - min_col) / (eps + max_col - min_col))
}

col_scale <- function(col_train,
                      spatz = 0.05,
                      col_test,
                      eps = 0.0) {
  # use only train to determine scale-parameter
  max_col = max(col_train) * (1 + spatz)
  min_col = min(col_train) * (1 - spatz)
  col_train = u_scale(col_train, min_col = min_col, max_col = max_col, eps)
  col_test = u_scale(col_test, min_col = min_col, max_col = max_col, eps)
  return (list(col_train = col_train, col_test = col_test))
}

load_data <- function(path,
                     split_num = 0,
                     spatz = 0.05,
                     x_scale = TRUE,
                     eps = 0) {
  idx_train <- read.table(paste0(path, '/data/index_train_', split_num, '.txt')) + 1 #We are R-Based
  idx_test <- read.table(paste0(path, '/data/index_test_', split_num, '.txt')) + 1 #We are R-Based
  y_col <- read.table(paste0(path, '/data/index_target.txt'))  + 1
  x_cols <- read.table(paste0(path, '/data/index_features.txt'))  + 1
  runs <- as.numeric(read.table(paste0(path, '/data/n_splits.txt')))
  dat <- as.matrix(read.table(paste0(path, '/data/data.txt')))
  X <- dat[, x_cols$V1]
  y <- dat[, y_col$V1]
  X_train <- X[idx_train$V1, ]
  y_train <- y[idx_train$V1]
  X_test <- X[idx_test$V1, ]
  y_test <- y[idx_test$V1]
  # determine scale parameter for y-values before normalization
  max_y <- max(y_train) * (1 + spatz)
  min_y <- min(y_train) * (1 - spatz)
  scale <- max_y - min_y
  # normalize y_train and y_test based y_train values

  if (x_scale == TRUE) {
    X_train <- as.matrix(X_train) # also in case of 1 x
    for (i in 1:ncol(X_train)) {
      # normalize x_train and x_test based x_train values
      X_s2 <- col_scale(
        col_train = X_train[, i],
        spatz = 0,
        col_test = X_test[, i],
        eps
      )
      X_train[, i] <- X_s2$col_train
      X_test[, i] <- X_s2$col_test
    }

  }
  return (
    list(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      y_test = y_test,
      runs = runs,
      scale = scale
    )
  )
}

get_data_wine <- function(path,
                         split_num = 0,
                         spatz = 0.05,
                         x_scale = FALSE) {
  name <- 'wine-quality-red'
  ret <- load_data(path, split_num, spatz, x_scale)
  ret$name <- name
  return (ret)
}

preproc <- function(vec,
                    labs = 1:6,
                    levs = 3:8,
                    retfac = FALSE) {
  tmp <- factor(vec,
                levels = levs,
                labels = labs,
                ordered = TRUE)
  mat <-
    model.matrix( ~ 0 + tmp, contrasts.arg = list(tmp = "contr.treatment"))
  if (retfac)
    return(tmp)
  return(mat)
}
