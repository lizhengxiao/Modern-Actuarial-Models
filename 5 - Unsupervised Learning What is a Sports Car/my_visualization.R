###############################################################
########## 加载数据
###############################################################
library(tsne)
library(umap)
library(kohonen)
dat = read.csv("D:\\Files\\Github\\ActuarialDataScience\\5 - Unsupervised Learning What is a Sports Car\\SportsCars.csv",sep=';')
dat1 = dat   
dat1$x1 = log(dat1$weight/dat1$max_power)
dat1$x2 = log(dat1$max_power/dat1$cubic_capacity)
dat1$x3 = log(dat1$max_torque)
dat1$x4 = log(dat1$max_engine_speed)
dat1$x5 = log(dat1$cubic_capacity)
dat1 = dat1[, c("x1","x2","x3","x4","x5")]
X_cent = dat1-colMeans(dat1)[col(dat1)]# 中心化
X_stan = X_cent/sqrt(colMeans(X_cent^2))[col(X_cent)]# 标准化
##########################################################################
#########  t SNE
##########################################################################
perp = 30 #困惑度
set.seed(100)
{t = proc.time()
    tsne = tsne(X_stan, k=2, initial_dim=ncol(X_stan), perplexity=perp)
proc.time()-t}
tsne1 = tsne[,c(2,1)]
plot(tsne1,col="blue",pch=20,ylab="component 1",xlab="component 2",
     main="t-SNE with perplexity 30")
points(tsne1[which(dat$tau<21),], col="green",pch=20)
points(tsne1[which(dat$tau<17),], col="red",pch=20)
##########################################################################
#########  UMAP
##########################################################################
umap.defaults
min_dist = c(0.1,0.5,0.9)
k = c(15,50,75,100)
sign = matrix(c(1,-1,-1,1,1,1,-1,-1),4,2,byrow = F)
par(mfrow=c(3,4))
for (i in 1:3){
    for (j in 1:4){
        umap.param = umap.defaults
        umap.param$n_components = 2
        umap.param$random_state = 100
        umap.param$min_dist = min_dist[i]
        umap.param$n_neighbors = k[j]
        umap = umap(X_stan, config=umap.param, method="naive")
        umap1 = matrix()
        umap$layout[,1] = sign[j,1]*umap$layout[,1]
        umap$layout[,2] = sign[j,2]*umap$layout[,2]
        umap1 = umap$layout[,c(2,1)]
        plot(umap1,col="blue",pch=20,
             ylab="component 1", xlab="component 2",
             main=list(paste("UMAP (k=",k[j],", min_dist= ",min_dist[i], ")",sep="")))
        points(umap1[which(dat$tau<21),], col="green",pch=20)
        points(umap1[which(dat$tau<17),], col="red",pch=20)
    }
}
##########################################################################
#########  Kohonen map
##########################################################################
p = ceiling(sqrt(5*sqrt(nrow(X_stan)))) #根据经验公式确定正方形的边长
set.seed(100)
som1 = som(as.matrix(X_stan),grid=somgrid(xdim=p,ydim=p,topo="rectangular"),rlen=500,dist.fcts="euclidean")
summary(som1)
head(som1$unit.classif,100) #各数据点的获胜神经元
tail(som1$codes[[1]]) #连接权重向量
set.seed(100)
som2 = som(as.matrix(X_stan),grid=somgrid(xdim=p,ydim=p,topo="hexagonal"),rlen=500,dist.fcts="euclidean")
par(mfrow=c(2,3))
plot(som1,c("changes"))
plot(som1,c("counts"), main="allocation counts to neurons")
dat$tau2 = dat$sports_car+as.integer(dat$tau<21)+1
plot(som1,c("mapping"),classif=predict(som1),col=c("blue","green","red")[dat$tau2], pch=19, main="allocation of cases to neurons")
plot(som2,c("changes"))
plot(som2,c("counts"), main="allocation counts to neurons")
dat$tau2 = dat$sports_car+as.integer(dat$tau<21)+1
plot(som2,c("mapping"),classif=predict(som2),col=c("blue","green","red")[dat$tau2], pch=19, main="allocation of cases to neurons")