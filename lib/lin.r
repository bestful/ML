MC_LINEAR <- T

lic.adaline <- function(xl, u, w){
  sign(w*t(u))
}

lic.grad()