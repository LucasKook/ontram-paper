# Script for reading UTKFace data from hdf5 into R
# 04.06.2020
# Lucas Kook


# Dependencies ------------------------------------------------------------

library(rhdf5) # From Bioconductor

# Functions ---------------------------------------------------------------

read_data <- function(mode, path = "/home/data/UTKFace/UTKFace/UTKFace.h5", img = TRUE) {
  dset <- paste0("_", mode)
  inm <- paste0("X", dset) # images
  nms <- paste0(c("age_group", "gender", "race",
                  paste0("x_", 1:10)), dset) # clinical data and response
  if (img)
    img <- read_img(path, inm)
  dat <- read_tab(path, nms)
  return(list(img, dat))
}

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
  train <- read_data("train")
}
