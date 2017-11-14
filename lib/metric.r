
mc.knn <- function(xl, u, k, sorted=FALSE, metric=norm){
  cols <- ncol(xl)
  rows <- nrow(xl)
  if(sorted != TRUE){
    umat <- matrix(rep(u, rows), rows, cols-1, byrow=TRUE)
    xl <- xl[order(metric(umat - xl[,1:(cols-1)] )),]
  }
  xl[which.max(table(xl[1:k,cols])), cols]
}

mc.kwnn <- function(xl, u, k, wf, sorted=FALSE, metric=norm){
  cols <- ncol(xl)
  rows <- nrow(xl)
  
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

mc.parzen <- function(xl, u, h, K, sorted=FALSE, metric=norm){
  cols <- ncol(xl)
  rows <- nrow(xl)
  
  umat <- matrix(rep(u, rows), rows, cols-1, byrow=TRUE)
  distances <- metric(umat - xl[,1:(cols-1)])
  if(sorted != TRUE){
    orderedIndexes <- order(distances)
    xl <- xl[orderedIndexes,]
    distances <- distances[orderedIndexes]
  }
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

mc.parzen.auto <- function(xl, u, k, K, sorted=FALSE, metric=norm){
  cols <- ncol(xl)
  rows <- nrow(xl)
  
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
