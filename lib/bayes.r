ML_BAYES <- T

N = function(x, m, cv) {
  n = length(x)
  x = as.matrix(x)
  arg = x-m
  1/sqrt(2*pi)**n/sqrt(det(cv))*exp(-0.5 * t(arg)%*%solve(cv)%*%arg)
}

bc.m <- function(xl){
  ncols <- ncol(xl)-1
  nrows <- nrow(xl)
  classes <- names(table(xl[,ncols+1]))
  cl_len <- length(classes)
  
  m = matrix(0, nrow=cl_len, ncol=ncols)
  row.names(m) <- classes

  for (i in classes) {
    xll <- xl[xl[,ncols+1]==i,]

    for (j in 1:ncols) {
      m[i,j] <- mean(xll[,j])
    }
  }
  m
}

bc.s <- function(xl){
  ncols <- ncol(xl)-1
  nrows <- nrow(xl)
  classes <- names(table(xl[,ncols+1]))
  cl_len <- length(classes)
  
  s = matrix(0, nrow=cl_len, ncol=ncols)
  row.names(s) <- classes
  
  for (i in classes) {
    xll = xl[xl[,ncols+1]==i,]
    
    for (j in 1:ncols) {
      s[i,j] = sqrt(var(xll[,j]))
    }
  }
  s
}

bc.apr <- function(xl){
  ncols <- ncol(xl)-1
  nrows <- nrow(xl)
  classes <- names(table(xl[,ncols+1]))
  cl_len <- length(classes)
  
  a <- c()
  for (i in classes) {
    a[i] = length(which(xl[,ncols+1] == i))
  }
  a/nrows
}

bc.cov <- function(xl){
  ncols <- ncol(xl)-1
  nrows <- nrow(xl)
  classes <- names(table(xl[,ncols+1]))

  cv = list()
  for (i in classes) {
    xll = xl[xl[,ncols+1]==i,1:ncols]
    cv[[i]] = cov(xll)
  }
  cv
}

bc.plugin <- function(xl, u, apr, m, cv){
  ncols <- ncol(xl)-1
  nrows <- nrow(xl)
  classes <- names(table(xl[,ncols+1]))
  cl_len <- length(classes)
  
  score = rep(0, cl_len)
  names(score) <- classes

  
  for (i in classes) {
    score[i] = apr[i] * N(u, m[i,], cv[[i]])
  }
  classes[which.max(score)]
}

bc.fisher <- function(xl, u, apr, m){
  ncols <- ncol(xl)-1
  nrows <- nrow(xl)
  classes <- names(table(xl[,ncols+1]))
  cl_len <- length(classes)
  xl[,ncols+1]<-"none"
  cv<-bc.cov(xl)
  cv<-(nrows-1)/(nrows-cl_len)*cv[["none"]]
  cvinv<-cv
  
  score = rep(0, cl_len)
  names(score) <- classes

  for (i in classes) {
    score[i] = log(apr[i]) - 1/2*t(m[i,])%*%cvinv%*%m[i,] + u%*%cvinv%*%m[i,]
  }
  classes[which.max(score)]
}
