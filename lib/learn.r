ML_LEARN <- T

learn.gamma <- function(f, xl, ...){
  cols <- ncol(xl)
  rows <- nrow(xl)
  g <- rep(0, rows)
  
  for(i in seq(1, rows, 1)){
    el <- xl[i,]
    class <- el[cols]
    el <- el[1:(cols-1)]
    while(class!=f(xl, el, g, ...)){
      g[i] <- g[i]+1
    }
  }
  g
}

learn.gamma.gen <- function(f, xl,  ...){
  g <- learn.gamma(f, xl,  ...)
  gen <- f
  formals(gen) <- formals(f)[-3]
  en <- environment(gen)
  en$g = g
  gen
}

learn.gamma.gen2 <- function(f, xl,  ...){
  gen <- learn.gamma.gen(f, xl, ...)
  formals(gen) <- formals(f)[-1]
  en <- environment(gen)
  en$f = f
  gen
}

learn.loo <- function(f, xl, lstOfX, ...){
  lst <- loo.list(f, xl, lstOfX, ...)
  i <- which.min(lst[,2])
  lst[i,1]
}