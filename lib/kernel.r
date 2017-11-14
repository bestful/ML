
ker.T <- function(r){
  if(abs(r)<=1)
    1-abs(r)
  else
    0
}

ker.P <- function(r){
  if(abs(r)<=1)
    1/2
  else
    0
}

ker.G <- function(r){
  (2*pi)^(1/2 * exp(-1/2 * r^2))
}

ker.E <- function(r){
  if(abs(r)<=1)
    3/4 * (1-r^2)
  else
    0
}

ker.Q <- function(r){
  if(abs(r)<=1)
    15/16*(1-r^2)^2
  else
    0
}