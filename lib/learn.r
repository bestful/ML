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
  gen <- fplo
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

learn.adaline <- function(xl, temp, lambda, iter){
  ncols <- ncol(xl)-1
  nrows <- nrow(xl)
  classes <- names(table(xl[,ncols+1]))
  w <- rep(0, ncols)
  
  
  Q <- sqrt(sum((w%*%t(xl[,1:ncols])-t(xl[,ncols+1]))**2))
  i<-0
  for(j in 1:iter){
    print(w)
    i<-(i+1)%%(nrows+1)
    x<-xl[i,1:ncols]
    y<-xl[,ncols+1]
    w <- w - temp*(t(w)*x - y)*x
    lines(c(0, -w[1]/w[2]), c(-w[1]/w[3], 0))
  }
  w
}
