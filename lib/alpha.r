

mc.stolp(xl, l0, delta1, delta0, f, ot, ...){
  cols <- ncol(xl)
  rows <- nrow(xl)
  
  for(u in xl){
    o <- ot(xl, u, ...)
    xla <- rbind(xla, xl[o>delta1,])
  }
  
  l <- 1
  while(l>=l0){
    err<-0
    xlaa <- c()
    for(u in xl){
      class <- f(xla, u, ...)
      if(class != u[cols]){
        if(ot(xla, u, ...)<delta0){
          xlaa <- rbind(xlaa, u)
        }
        err <-- err+1
      }
    }
    xla <- unique(rbind(xla, xlaa))
    l <- err/cols
  }
  
}
