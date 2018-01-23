MC_LINEAR <- T

lin.clsfy <- function(xl, u, w){
  sign(w*t(data.frame(-1, u)))
}


lin.SG <- function(xl, L, upd, lam=0.01, maxIter=1000, callback=function(...){} ) {
  ncols <- ncol(xl)-1
  nrows <- nrow(xl)
  
  w = runif(ncols, -1 / (2 * ncols), 1 / (2 * ncols))
  
  Q <- 0
  for (i in 1:nrows) {
    M <- sum(w * xl[i,1:ncols]) * xl[i, ncols+1]
    Q <- Q + L(M)
  }
  Q.prev <- Q
  
  x <- xl[, 1:ncols]
  y <- xl[, ncols+1]
  it <- 1
  
  while(it<maxIter || abs(Q - Q.prev)/Q < 1e-4) {
    i <- sample(1:nrows, 1)
    xi <- x[i,]
    yi <- y[i]
    M <- sum(w * xi) * yi
    err <- L(M)
    temp <- 1 / it
    w <- upd(w, temp, xi, yi)
 #   w <- w/norm(w)
    Q.prev <- Q
    Q <- (1 - lam) * Q + lam * err
    callback(w, x)
    it <- it+1
  }
  
  w
}

learn.lin.adaline <- function(xl, ...){
  lin.SG(xl, 
         function(M){ 
           (M-1)**2 
         }, 
         function(w, temp, xi, yi){
           w - temp*(sum(w*xi)-yi)*xi
         }, ...)
}

learn.lin.perceptron <- function(xl, ...){
  lin.SG(xl, 
         function(M){ 
           if(M<0)
             -M
           else
             0
         }, 
         function(w, temp, xi, yi){
           if(sum(w*xi)*yi < 0)
             w + temp*xi*yi
           else
             w
         }, ...)
}

learn.lin.logistic <- function(xl, ...){
  lin.SG(xl, 
              function(M){ 
                log2(1+exp(-M))
              }, 
              function(w, temp, xi, yi){
                w + temp*yi*xi*(1/(1+exp(sum(w*xi)*yi)))
              }, ...)
}

lin.adaline <- function(xl, ...){
  w <- learn.lin.adaline(data.frame(-1, xl), ...)
  gen <- lin.clsfy
  formals(gen) <- formals(gen)[-3]
  en <- environment(gen)
  en$w = w
  gen
}

lin.perceptron <- function(xl, ...){
  w <- learn.lin.perceptron(data.frame(-1, xl), ...)
  gen <- lin.clsfy
  formals(gen) <- formals(gen)[-3]
  en <- environment(gen)
  en$w = w
  gen
}


lin.logistic <- function(xl, ...){
  w <- learn.lin.logistic(data.frame(-1, xl), ...)
  gen <- lin.clsfy
  formals(gen) <- formals(gen)[-3]
  en <- environment(gen)
  en$w = w
  gen
}

