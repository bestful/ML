source("../lib/common.r")
source("../lib/metric.r")
source("../lib/kernel.r")

p <- loo.list(mc.parzen, sel, seq(from=0.1, to=5, by=0.1), ker.T)
plot(p, type="l", xlab="k in knn", ylab="errors using loo")


#init
sel<-iris[,3:5]
pl<-sel[,1:2]
mi <- c(1, 0.1)
ma <- c(7, 2.5)
acc <- 3
colors <- c(setosa="red", versicolor="green", virginica="blue")
par("lwd")

#1NN lattice with LOO
plot(pl, col=colors[sel$Species], main="1NN", pch=19)
k <- 1
lattice(mc.knn, sel, colors, mi, ma, 0.1, k)
legend(mi[1], ma[2], legend=paste("LOO =", round(loo(mc.knn, sel, 1), digits=3)))

#LOO(k) in KNN
p<-loo.list(mc.knn, sel, 1:50)
plot(p, type="l", xlab="k", ylab="error", main="LOO KNN")
legend(0, 0.066, legend=paste("LOO(6) =", round(loo(mc.knn, sel, 6), digits=3)))

#6NN lattice with LOO
plot(pl, col=colors[sel$Species], main="6NN", pch=19)
k <- 6
lattice(mc.knn, sel, colors, mi, ma, 0.1, k)
legend(mi[1], ma[2], legend=paste("LOO =", round(loo(mc.knn, sel, k), digits=3)))

#LOO in kwnn
p<-loo.list(mc.kwnn, sel, 1:50, mc.wlin)
plot(p, type="l", xlab="k", ylab="error", main="LOO KWNN")
legend(0, 0.066, legend=paste("LOO(6) =", round(loo(mc.knn, sel, 6), digits=3)))

#6WNN lattice with LOO
plot(pl, col=colors[sel$Species], main="6WNN", pch=19)
k <- 6
lattice(mc.kwnn, sel, colors, mi, ma, 0.1, k, mc.wlin)
legend(mi[1], ma[2], legend=paste("LOO =", round(loo(mc.knn, sel, k), digits=3)))

#LOO in parzen
hs<-seq(from=0.1, to=2, by=0.1)
p<-loo.list(mc.parzen, sel, hs, ker.T)
plot(p, type="l", xlab="k", ylab="error", main="LOO PARZEN WINDOW", ylim=c(0, 0.2))
lines(loo.list(mc.parzen, sel, hs, ker.P),  col = "red")
lines(loo.list(mc.parzen, sel, hs, ker.E),  col = "green")
lines(loo.list(mc.parzen, sel, hs, ker.G),  col = "blue")
lines(loo.list(mc.parzen, sel, hs, ker.Q),  col = "grey")
legend(1.8, 0.2, legend=c("K=T", "K=P", "K=E", "K=G", "K=Q"), lty=5, col=c("black", "red", "green", "blue", "grey") )

#LOO in parzen auto
hs<-1:20
p<-loo.list(mc.parzen.auto, sel, hs, ker.T)
plot(p, type="l", xlab="k", ylab="error", main="LOO PARZEN WINDOW", ylim=c(0, 0.2))
lines(loo.list(mc.parzen.auto, sel, hs, ker.P),  col = "red")
lines(loo.list(mc.parzen.auto, sel, hs, ker.E),  col = "green")
lines(loo.list(mc.parzen.auto, sel, hs, ker.G),  col = "blue")
lines(loo.list(mc.parzen.auto, sel, hs, ker.Q),  col = "grey")
legend(1.8, 0.2, legend=c("K=T", "K=P", "K=E", "K=G", "K=Q"), lty=5, col=c("black", "red", "green", "blue", "grey") )

#6WNN lattice with LOO
plot(pl, col=colors[sel$Species], main="6WNN", pch=19)
k <- 6
lattice(mc.kwnn, sel, colors, mi, ma, 0.1, k, mc.wlin)
legend(mi[1], ma[2], legend=paste("LOO =", round(loo(mc.knn, sel, k), digits=3)))

##
plot(pl, col=colors[sel$Species], pch=19)
k <- 6
w <- apply(data.frame(k, 1:k), 1, mc.w )
loo(mc.knn, sel, k)
lattice(mc.parzen.auto, sel, colors, c(1,0.1), c(7,2.5), 0.1, k, ker.T)
lattice(mc.kwnn, sel, colors, c(1,0.1), c(7,2.5), 0.1, k, mc.w)


xl<-sel
cols <- ncol(xl)
rows <- nrow(xl)
u<-c(1.4, 0.2)
umat <- matrix(rep(u, rows), rows, cols-1, byrow=TRUE)
vmat<-xl[,1:(cols-1)]
distances <- norm(umat - xl[,1:(cols-1)])

  orderedIndexes <- order(distances)
  xl <- xl[orderedIndexes,]
  distances <- distances[orderedIndexes]

xl <- xl[,cols]
wp <- distances[1:k]/distances[k+1]