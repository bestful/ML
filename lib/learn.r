ML_LEARN <- T

learn.gamma <- function(f, xl, by, ...){
  cols <- ncol(xl)
  rows <- nrow(xl)
  g <- rep(0, rows)
  
  for(i in seq(1, rows, 1)){
    el <- xl[i,]
    class <- el[cols]
    while(class!=f(xl, el, g, ...)){
      g[i] <- g[i]+by
    }
  }
  g
}

learn.loo <- function(f, xl, lstOfX, ...){
  
}