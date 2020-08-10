# Script for reading DR data from hdf5 into R
# 21.05.2020
# Lucas Kook


# Dependencies ------------------------------------------------------------

library(rhdf5) # From Bioconductor

# Names and path ----------------------------------------------------------

path_train <- "/home/data/DR_data/train_resized.h5"
path_valid <- "/home/data/DR_data/valid_resized.h5"
path_test <- "/home/data/DR_data/test_resized.h5"
inm <- "x" # images
nms <- c("y", paste0("x_", 1:10)) # clinical data and response

# Functions ---------------------------------------------------------------

read_img <- function(path, name) {
  img <- h5read(path, name)
  ret <- aperm(img, 4:1)
  return(ret)
}

read_tab <- function(path, names) {
  dat <- sapply(names, function(nm) {h5read(path, nm)})
  ret <- data.frame(dat)
  return(ret)
}

# Read data ---------------------------------------------------------------

if (FALSE) {
  img_train <- read_img(path_train, inm)
  dat_train <- read_tab(path_train, nms)

  img_valid <- read_img(path_valid, inm)
  dat_valid <- read_tab(path_valid, nms)

  img_test <- read_img(path_test, inm)
  dat_test <- read_tab(path_test, nms)
}
