ML_COMMON <- T

norm <- function(x){
  r <- apply(x^2, 1, sum)
  sqrt(r)
}


lattice <- function(f, xl, color, mi, ma, acc, ...){
  for(x in seq(mi[1], ma[1], acc)){
    for(y in seq(mi[2], ma[2], acc)){
      u <- c(x, y)
      class <- f(xl, u, ...)
      points(x, y, col=color[class], pch="+")
    }
  }
}

lattice.without <- function(f, color, mi, ma, acc, ...){
  for(x in seq(mi[1], ma[1], acc)){
    for(y in seq(mi[2], ma[2], acc)){
      u <- c(x, y)
      class <- f(u, ...)
      points(x, y, col=color[class], pch=21)
    }
  }
}

loo <-function(f, xl, ...){
  cols <- ncol(xl)
  rows <- nrow(xl)
  acc <- 0
  
  for(i in 1:rows){
    xli <- xl[-i,]
    u <- unname(unlist(xl[i, 1:(cols-1)]))
    class <- f(xli, u, ...)
    if(xl[i,cols] == class){
      acc <- acc+1
    }
  }
  
  1-acc/rows
}

loo.list <- function(f, xl, lstOfX,...){
  p <- c()
  for(x in lstOfX){
    y <- loo(f, xl, x, ...)
    p <- rbind(p, c(x,y))
  }
  p
}
