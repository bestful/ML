ML_METRIC <- T

if(ML_COMMON == F)
  source("common.r")

mc.knn <- function(xl, u, k, metric=norm, sorted=FALSE){
  cols <- ncol(xl)
  rows <- nrow(xl)
  u <- unname(unlist(u))
  if(sorted != TRUE){
    umat <- matrix(rep(u, rows), rows, cols-1, byrow=TRUE)
    xl <- xl[order(metric(umat - xl[,1:(cols-1)] )),]
  }
  xl <- xl[1:k,cols]
  
  classes <- names(table(xl))
  score <- rep(0, length(classes))
  i <- 1
  
  for(el in xl){
    class <- xl[i]
    score[class] <- score[class]+1
    i <- i+1
  }
  
  classes[which.max(score)]
}

mc.wlin <- function(el){
  k <- el[1]
  i <- el[2]
  (k+1-i)/k
}

mc.kwnn <- function(xl, u, k, wf, metric=norm, sorted=FALSE){
  cols <- ncol(xl)
  rows <- nrow(xl)
  u <- unname(unlist(u))
  
  if(sorted != TRUE){
    umat <- matrix(rep(u, rows), rows, cols-1, byrow=TRUE)
    w <- apply(data.frame(k, 1:k), 1, wf)
    w <- matrix(w, length(w), 1)
    xl <- xl[order(metric(umat - xl[,1:(cols-1)] )),]
  }
  xl <- xl[1:k,cols]
  
  classes <- names(table(xl))
  score <- rep(0, length(classes))
  i <- 1
  
  for(el in xl){
    class <- xl[i]
    score[class] <- score[class]+w[i]
    i <- i+1
  }
  
  classes[which.max(score)]
}

mc.parzen <- function(xl, u, h, K, metric=norm){
  cols <- ncol(xl)
  rows <- nrow(xl)
  u <- unname(unlist(u))
  
  umat <- matrix(rep(u, rows), rows, cols-1, byrow=TRUE)
  distances <- metric(umat - xl[,1:(cols-1)])
  xl <- xl[,cols]
  wp <- distances/h
  
  classes <- c(names(table(xl)), "none")
  score <- rep(0, length(classes))
  score[length(classes)]<-0.00001
  i <- 1
  
  for(el in xl){
    class <- xl[i]
    score[class] <- score[class]+K(wp[i])
    i <- i+1
  }
  
  classes[which.max(score)]
}

mc.parzen.auto <- function(xl, u, k, K, metric=norm, sorted=FALSE){
  cols <- ncol(xl)
  rows <- nrow(xl)
  u <- unname(unlist(u))
  
  umat <- matrix(rep(u, rows), rows, cols-1, byrow=TRUE)
  distances <- metric(umat - xl[,1:(cols-1)])
  if(sorted != TRUE){
    orderedIndexes <- order(distances)
    xl <- xl[orderedIndexes,]
    distances <- distances[orderedIndexes]
  }
  xl <- xl[1:k,cols]
  h<-distances[k+1]
  if(h==0)
    h<-0.01
  
  wp <- distances[1:k]/h
  
  classes <- c(names(table(xl)), "none")
  score <- rep(0, length(classes))
  score[length(classes)]<-0.00001
  i <- 1
  
  for(el in xl){
    class <- xl[i]
    score[class] <- score[class]+K(wp[i])
    i <- i+1
  }
  
  classes[which.max(score)]
}

mc.poten <- function(xl, u, g, h, K, metric=norm){
  cols <- ncol(xl)
  rows <- nrow(xl)
  u <- unname(unlist(u))
  
  umat <- matrix(rep(u, rows), rows, cols-1, byrow=TRUE)
  distances <- metric(umat - xl[,1:(cols-1)])
  xl <- xl[,cols]
  wp <- distances/h
  
  classes <- c(names(table(xl)), "none")
  score <- rep(0, length(classes))
  score[length(classes)]<-0.00001
  i <- 1
  
  for(el in xl){
    class <- xl[i]
    score[class] <- score[class]+g[i]*K(wp[i])
    i <- i+1
  }
  
  classes[which.max(score)]
}

mc.knn.margin = function(xl, u, y, k, metric=norm) {
  cols <- ncol(xl)
  rows <- nrow(xl)
  u <- unname(unlist(u))

    umat <- matrix(rep(u, rows), rows, cols-1, byrow=TRUE)
    xl <- xl[order(metric(umat - xl[,1:(cols-1)] )),]

  xl <- xl[1:k,cols]
  
  classes <- names(table(xl))
  score <- rep(0, length(classes))
  i <- 1
  
  for(el in xl){
    class <- xl[i]
    score[class] <- score[class]+1
    i <- i+1
  }
  clsy <- score[y]
  score[y] <- 0
  clsy - which.max(score)
}

mc.stolp <- function(fm, xl, delta, l0, ...){
  cols <- ncol(xl)
  rows <- nrow(xl)
  
  g = rep(T, times=rows)
  for (i in 1:rows) {
    if (fm(xl, xl[i,1:(cols-1)], xl[i,cols], k) < delta) {
      g[i] = F
    }
  }
  xl = xl[g,]
  rows <- nrow(xl)
  
  classes <- names(table(xl[cols]))
  score <- rep(0, length(classes))
  els <- rep(0, length(classes))
  for(i in 1:rows){
    class <- xl[i,cols]
    M <- fm(xl, xl[i,1:(cols-1)], class, ...)
    if(score[class]<M){
      els[class] <- i
      score[class] <- M
    }
  }
  inomega = rep(F, times=rows)
  for(el in els){
    inomega[el] <- T
  }
  omega <- xl[inomega,]
  xl_omega <- xl[!inomega,]
  
  while(nrow(omega)!=nrow(xl)){
    l <- nrow(xl_omega)
    g<-rep(F, times=l)
    for(i in 1:l){
      if(fm(xl_omega, xl_omega[i,1:(cols-1)], xl_omega[i, cols], ...)<0){
        g[i] = T
      }
    }
    E <- xl_omega[g,]
    lE <- nrow(E)
    if(lE < l0){
      break
    }
    
    minM <- 0
    minI <- 0
    for(i in 1:lE) {
      M <- fm(E, E[i, 1:(cols-1)], E[i, cols], ...) 
      if(M<minM){
        minI <- i
      }
    }
    
    omega <- rbind(omega, E[minI,])
    g<-rep(T, times=l)
    for(i in 1:l){
      if(all(xl_omega[i,]==E[minI,])){
        g[i] = F
      }
      i <- i+1
    }
    xl_omega <- xl_omega[g,]
  }
  omega
}
