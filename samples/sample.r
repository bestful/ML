source("../lib/common.r")
source("../lib/metric.r")
source("../lib/kernel.r")
source("../lib/learn.r")
source("../lib/bayes.r")
source("../lib/linear.r")

#init
sel<-iris[,3:5]
pl<-sel[,1:2]
mi <- c(1, 0.1)
ma <- c(7, 2.5)
acc <- 3
colors <- c(setosa="red", versicolor="green", virginica="blue")
par("lwd")
w<-1000
h<-500

#1NN lattice with LOO
png('1nn.png', width=w, height=h)
plot(pl, col=colors[sel$Species], main="1NN", pch=19)
k <- 1
lattice(mc.knn, sel, colors, mi, ma, 0.1, k)
legend(mi[1], ma[2], legend=paste("LOO =", round(loo(mc.knn, sel, 1), digits=3)))
dev.off()

#LOO(k) in KNN
png('loo_knn.png', width=w, height=h)
p<-loo.list(mc.knn, sel, 1:50)
plot(p, type="l", xlab="k", ylab="error", main="LOO KNN")
legend(0, 0.066, legend=paste("LOO(6) =", round(loo(mc.knn, sel, 6), digits=3)))
dev.off()

#6NN lattice with LOO
png('6nn.png', width=w, height=h)
plot(pl, col=colors[sel$Species], main="6NN", pch=19)
k <- 6
lattice(mc.knn, sel, colors, mi, ma, 0.1, k)
legend(mi[1], ma[2], legend=paste("LOO =", round(loo(mc.knn, sel, k), digits=3)))
dev.off()

#LOO in kwnn
png('loo_kwnn.png', width=w, height=h)
p<-loo.list(mc.kwnn, sel, 1:50, mc.wlin)
plot(p, type="l", xlab="k", ylab="error", main="LOO KWNN")
legend(0, 0.066, legend=paste("LOO(6) =", round(loo(mc.kwnn, sel, 6, mc.wlin), digits=3)))
dev.off()

#6WNN lattice with LOO
png('6wnn.png', width=w, height=h)
plot(pl, col=colors[sel$Species], main="6WNN", pch=19)
k <- 6
lattice(mc.kwnn, sel, colors, mi, ma, 0.1, k, mc.wlin)
legend(mi[1], ma[2], legend=paste("LOO =", round(loo(mc.knn, sel, k), digits=3)))
dev.off()

#Lattice parzen
png('loo_parzen.png', width=800, height=800)
par(mfrow=c(3, 2))
hs<-seq(from=0.1, to=2, by=0.1)
p<-loo.list(mc.parzen, sel, hs, ker.T)
plot(p, type="l", xlab="k", ylab="error", main="LOO PARZEN WINDOW K=Triangular", ylim=c(0, 0.2))
plot(loo.list(mc.parzen, sel, hs, ker.P),
     type="l", xlab="k", ylab="error", main="LOO PARZEN WINDOW K=Pramougolnoe", ylim=c(0, 0.2))
plot(loo.list(mc.parzen, sel, hs, ker.E),
     type="l", xlab="k", ylab="error", main="LOO PARZEN WINDOW K=Epanechnikova", ylim=c(0, 0.2))
plot(loo.list(mc.parzen, sel, hs, ker.G),
     type="l", xlab="k", ylab="error", main="LOO PARZEN WINDOW K=Gaussian", ylim=c(0, 0.2))
plot(loo.list(mc.parzen, sel, hs, ker.Q),
     type="l", xlab="k", ylab="error", main="LOO PARZEN WINDOW K=Quarticheskoe", ylim=c(0, 0.2))
dev.off()

#LOO in parzen auto
png('loo_parzen_auto.png', width=800, height=800)
par(mfrow=c(3, 2))
hs<-1:20
p<-loo.list(mc.parzen.auto, sel, hs, ker.T)
plot(p, type="l", xlab="h", ylab="error", main="LOO PARZEN WINDOW AUTO K=Triangular", ylim=c(0, 0.2))
plot(loo.list(mc.parzen.auto, sel, hs, ker.P),
     type="l", xlab="h", ylab="error", main="LOO PARZEN WINDOW AUTO K=Pramougolnoe", ylim=c(0, 0.2))
plot(loo.list(mc.parzen.auto, sel, hs, ker.E),
     type="l", xlab="h", ylab="error", main="LOO PARZEN WINDOW AUTO K=Epanechnikova", ylim=c(0, 0.2))
plot(loo.list(mc.parzen.auto, sel, hs, ker.G),
     type="l", xlab="h", ylab="error", main="LOO PARZEN WINDOW AUTO K=Gaussian", ylim=c(0, 0.2))
plot(loo.list(mc.parzen.auto, sel, hs, ker.Q),
     type="l", xlab="h", ylab="error", main="LOO PARZEN WINDOW AUTO K=Quarticheskoe", ylim=c(0, 0.2))
dev.off()
par(mfrow=c(1, 1))

#lattice poten
png('parzen.png', width=w, height=h)
plot(pl, col=colors[sel$Species], pch=19, main="Parzen K=Triangular, h=0.4")
lattice(mc.parzen, sel, colors, mi, ma, 0.1, 0.4, ker.T)
dev.off()

#lattice poten
png('poten.png', width=w, height=h)
plot(pl, col=colors[sel$Species], pch=19, main="Poten K=Triangular, h=0.4")
lattice(mc.poten, sel, colors, mi, ma, 0.1, learn.gamma(mc.poten, sel, 0.4, ker.T), 0.4, ker.T)
dev.off()

#stolp
png('stolp.png', width=w, height=h)
plot(pl, col=colors[sel$Species], pch=19, main="STOLP in 2nn, delta = -2, l0 = 3")
s<- mc.stolp(mc.knn.margin, sel, -2, 3, 2)
points(s, col=colors[s$Species], cex=2, pch=19)
dev.off()

png('plugin.png', width=w, height=h)
plot(pl, col=colors[sel$Species], pch=19, main="Plugin")
lattice(bc.plugin, sel, colors, mi, ma, 0.1, bc.apr(sel), bc.m(sel), bc.cov(sel))
legend(mi[1], ma[2], legend=paste("LOO =", round(loo(bc.plugin, sel, bc.apr(sel), bc.m(sel), bc.cov(sel)), digits=3)))

dev.off()

png('fisher.png', width=w, height=h)
plot(pl, col=colors[sel$Species], pch=19, main="Fisher")
lattice(bc.fisher, sel, colors, mi, ma, 0.1, bc.apr(sel), bc.m(sel))
legend(mi[1], ma[2], legend=paste("LOO =", round(loo(bc.fisher, sel, bc.apr(sel), bc.m(sel)), digits=3)))

dev.off()


png('adaline.png', width=w, height=h)
lcolors <- c("red", "green")
names(lcolors)<-c(-1,1)
lsel<-sel[sel$Species!="virginica",]
lsel[,3] <- as.vector(lsel$Species == "setosa")*2-1
plot(lsel[,1:2], col=lcolors[sel$Species], pch=19, main="Adaline t=0.05, 100 iterations")
w<-learn.adaline(data.frame(-1, lsel), 0.05,0.05,100)
lines(c(0, -w[1]/w[2]), c(-w[1]/w[3], 0), col="blue")

dev.off()
