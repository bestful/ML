# ML
# ����������� ��������� �������������
**����������� ������ ��������** -- ������, ���������� �� ������� �������� ��������.

**_����� ��������_** �������� ������� ���������� ![](http://latex.codecogs.com/svg.latex?%5Clarge%20%5Crho%3A%20%28X%20%5Ctimes%20X%29%20%5Crightarrow%20%5Cmathbb%7BR%7D). ��� ������ ���������� ����� ���������, ��� ������ ������� ������ ���� �� �����.

����������� ��������� ������������� ��������� �� **_�������� ������������_**: ������ �������� ������������� ������ ������.

����������� ��������� ������������� � ��������� �������� *Xl* ������� ������ *u* � ���� ������ *y*, ��� �������� **��������� ��� ��������� ��������� �������� ![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29) ����������**:

![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29%20%3D%20%5Csum_%7Bi%20%3A%20y_%7Bu%7D%5E%7B%28i%29%7D%20%3D%20y%7D%20w%28i%2C%20u%29%20%5Crightarrow%20max)

, ��� ������� ������� *w(i, u)* ��������� ������� �������� *i*-�� ������ ��� ������������� ������� *u*.

������� ![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29) ���������� **_������� �������� ������� u � ������ y_**. ������� ��������� ������� ������� *w(i, u)* ����� �������� ��������� ����������� ��������������.

��� ������ ����������� ���������� ��� ������� �� ��������������� ���� ����������� ���������� ������������ **LOO -- leave-one-out** *(�������� ����������� ��������)*, ������� ������� � ���������: 

1. ��������� ������� *x(i)* �� ������� *Xl* �� ������, ��������� ����� ������� ��� ������� *x(i)* (������ � *Xl_1*).
2. ��������� �������� �� ������� *u*, ������� ����� ����������������, �� ������� *Xl_1*.
3. ������� ���������� *Q* (���������� ������, ���������� *Q = 0*) �, ����� �������� ���������, *Q = Q + 1*.
4. ����� ��� ������� *x(i)* ����� ���������, ��������� *LOO = Q / l* (*l* -- ���������� �������� �������).

��� ����������� �������� LOO ������� ����������� �������� ���������.

## �������� k ��������� ������� (knn)


��� ������ �������� ������� _u_ � ������ _y_ �������� ���������� ���������
�������:
![](http://latex.codecogs.com/svg.latex?%5Clarge%20W%28i%2C%20u%29%20%3D%20%5Bi%20%5Cleq%20k%5D)
, ��� _i_ ���������� ������� ������ �� ���������� � ����� _u_.


�������� �������� _k_ ��������� ������� � ����������
��� �����, ������� ����� ��������� ����������� ������� ���������� ���.

���������� �������� ����������� ��������� �������:
```
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
```

![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/1nn.png)
��� k=1 �������� LOO = 0.047
� ������ ��������� LOO ��� ��������� k
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_knn.png)
��������� ��������� �������� ��� k=6
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/6nn.png)
��������� �� ������������ � ����������

### ������������:
- �������� ����������.
- ��� *k*, ����������� ����� ������������, �������� "�������" ��������������.

### ����������:
- ����� ������� ��� �������.
- ��� *k = 1* �������������� � ������������ (*��������* -- ��������, ������� �������� ��������� ������ ������), ���������� ���� ���� ������ ����������������� ������� � ���������� ��� �������, ��� �������� �� �������� ���������, ����.
- ��� *k = l* �������� �������� ��������� �������� � ����������� � ���������.
- ������������ ����� �������� � *counts* ����� ����������� � ���������� ������� ������������.
- "�������" ����� ����������.
- �����, ���������� ����� �������� ���������, �� ��� ����� �����������.

## �������� k ��������� ��������� ������� (kwnn)
������� ��������� ������� *Xl*, ��������� �� �������� *x(i), i = 1, ..., l* (� ����������� ��������� ������������ ������� ������ ������).
������ �������� ������������� ������� ������ *u* � ���� ������ *y*, � �������� ����������� ����� ����� *w_i* ��� ��������� *k* ������� *x(u_i)*.

��� ������ �������� ����������������� ������� *u* � ������ *y* **�������� kwnn** ���������� ��������� �������:

![](http://latex.codecogs.com/svg.latex?%5Clarge%20W%28i%2C%20u%29%20%3D%20%5Bi%20%5Cleq%20k%5D%20w%28i%29) , ��� *i* -- ������� ������ �� ���������� � ����������������� ������� *u*, � *w(i)* -- ������ ��������� ������� ����, ����� ����� i-�� ������ � �������������.

� ����������� ��������� ������������ ������� ������� ����: ![](https://latex.codecogs.com/gif.latex?w%28i%29%20%3D%20q%5Ei%2C%20q%20%5Cepsilon%20%280%2C%201%29)

���������� ������� �������:

``` R
mc.wlin <- function(el){
  k <- el[1]
  i <- el[2]
  (k+1-i)/k
}

```

���������� ��������������:
``` R
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
```
������� ������ ����������� k � kwnn ����� LOO
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_kwnn.png)
�������� �������� ����������� k ������. 
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/6wnn.png)
��� kwnn �����/���� knn?
- �� �� �����, ��� � knn
- ���� �������� ����������� k.
- ����� �������� �� ��������.

## ����� ������������� ����
������� ��������� ������� *Xl*, ��������� �� �������� *x(i), i = 1, ..., l* (� ����������� ��������� ������������ ������� ������ ������). � ������ ��������� ������� ������� *w_i* ������������ ��� ������� **�� ���������� ����� ���������������� �������� *u* � ��� �������� *x(u_i), i = 1, ..., l*, � �� �� ����� ������ *i***, ��� ���� � ������� kNN.

��� ������ �������� ����������������� ������� *u* � ������ *y* **����� ������������� ����** ���������� ��������� �������:

![](https://latex.codecogs.com/gif.latex?W%28i%2C%20u%29%20%3D%20K%5Cleft%20%28%20%5Cfrac%7B%5Crho%20%28u%2C%20x_%7Bu%7D%5E%7Bi%7D%29%7D%7Bh%7D%20%5Cright%20%29) , ��� *K(z)* -- ������� ���� (�� ������������ �� 0 �� �������������), � *h* -- ������ ���� (���� -- ����������� ����������� ����������������� ������� *u* ������� *h*).

��������������� 5 ����:
- ������������� 

![](https://latex.codecogs.com/gif.latex?K%28z%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5B%7Cz%7C%20%3C%3D%201%5D)
- ����������� 

![](https://latex.codecogs.com/gif.latex?K%28z%29%20%3D%20%281%20-%20%7Cz%7C%29%5B%7Cz%7C%20%3C%3D%201%5D)
- ������������ 

![](https://latex.codecogs.com/gif.latex?K%28z%29%20%3D%20%5Cfrac%7B15%7D%7B16%7D*%281%20-%20z%5E%7B2%7D%29%5B%7Cz%7C%20%3C%3D%201%5D)
- ����������� 

![](https://latex.codecogs.com/gif.latex?K%28z%29%20%3D%20%28%282%5Cpi%29%5E%7B%5E%7B%28%5Cfrac%7B-1%7D%7B2%7D%7D%29%7D%29*e%5E%7B%28%5Cfrac%7B-z%5E2%7D%7B2%7D%29%7D)
- ������������ 


![](https://latex.codecogs.com/gif.latex?K%28z%29%20%3D%20%5Cfrac%7B3%7D%7B4%7D*%281%20-%20z%5E2%29%5B%7Cz%7C%20%3C%3D%201%5D)

���������� ��������������:
``` R
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
```
������� ����������� ���:
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_parzen.png)
������ ������� ����� ������ � ������ ������� ���. ������� ����������� ���� � ����������� h = 0.4.
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/parzen.png)
��������� �� ����������� � ����������:

### �����
- ������� �������� ������������� ��� ��������� ���������� _h_
- ��� ����� � ���������� ����������� ����� �������

### ������
- ���������� ������� ��� ������� �������
- �������� ��������� _h_ ���������� ��������� ��������������, ��������
��������� ������������ �����
- ���� �� ���� ����� �� ������ � ������ _h_, �������� �� �������� ��
���������������� (�� ��������� ��� ������������ ����)

## ����� ������������� ���� � ���������� �����
�� �� �����, ��� � ������� ��������, ������ ������ ��� ��� ����������� � ����������� �� ���������� �� k ������. 
��������� LOO ��� ��������� ����:
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_parzen_auto.png)
������������� � ����������� ���� � ��������� k ���� ��������� �����, ��� � ���������� �����.
� ����� ������ �������������, ��������:
- ������������� ���� -> knn
- ����������� ���� -> kwnn c �������� w[i]
- ����������� ���� -> kwnn c� ����������� w[i]

## ����� ������������� �������

������� ��������� ������� *Xl*, ��������� �� �������� *x(i), i = 1, ..., l* (� ����������� ��������� ������������ ������� ������ ������). � ������ ��������� ������� ������� *w_i* ������������ ��� ������� �� ���������� ����� ���������������� �������� *u* � ��� �������� *x(u_i), i = 1, ..., l*, ��� � � **������ ������������� ����**.

��� ������ �������� ����������������� ������� *u* � ������ *y* **����� ������������� �������** ���������� ��������� �������:

![](https://latex.codecogs.com/gif.latex?W%28i%2C%20u%29%20%3D%20%5Cgamma_%7Bi%7D*K%5Cleft%20%28%20%5Cfrac%7B%5Crho%20%28u%2C%20x_%7Bu%7D%5E%7Bi%7D%29%7D%7Bh_i%7D%20%5Cright%20%29%2C%20%5Cgamma_%7Bi%7D%20%5Cgeqslant%200%2C%20h_i%20%3E%200) 

**�������� ����:** *����������* ���������� �������� ������� ������� *x_i* ��� �������������. �������, ��� ������� ����������� *h* �������� �������. �������� ��������� ������ ���������� ![](https://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D). 

���������� �����������:
``` R
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
```

���������� ���������:
``` R
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
```

**������ ������� ��������� ������������� ������� �� ���������� ����������:** ����� "����" ������������� � ��������� �������, � �� � ����������������.

����� �������������:
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/poten.png)

### �����:
- ��������� ������� �� _2l_ ����������

### ������:
- ���������� ������� ��� ������� �������
- ��������� _h_ ���������� ��������� ��������������, ��������
� �� ������� �� ��������� �������
- ���� �� ���� ����� �� ������ � ������ _h_, �������� �� �������� ��
���������������� (�� ��������� ��� ������������ ����)
- �������� ��������
- ������� ����� ����������� ���������
- �������������� ����� ������ (��� ��������� ������ ������ ����� ������
����������� ����������)

## STOLP
� ����������...
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/stolp.png)

# ����������� ��������� �������������
����������� ��������� ������������� �������� �� �������� ��������� ������������� �����������. ��� ����������������� ������� ����������� ��������� ������������� 
- **_������� �������������_** ������� ![](http://latex.codecogs.com/gif.latex?%5Cinline%20p%28x%7Cy%29%20%3D%20p_y%28x%29)
-  ***��������� �����������*** ������� ![](http://latex.codecogs.com/gif.latex?%5Cinline%20P_y)

�� ��� ����������� ***������������� �����������*** - ![](http://latex.codecogs.com/gif.latex?p%20%5Cleft%20%5C%7By%7Cx%20%5Cright%20%5C%7D%20%3D%20P_yp_y%28x%29)
������ ��������� � ������ � ������������ ������������� ������������.

*������ �������������* - �������� �������� ![](http://latex.codecogs.com/gif.latex?%5Cinline%20a%3A%5C%3B%20X%5Cto%20Y), ��������� ���������������� ������������ ������ ![](http://latex.codecogs.com/gif.latex?%5Cinline%20x%20%5Cin%20X).  

1)  ***���������� �������������� ��� ��������� ����������***  
![](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clambda_y) - ����� �� ������������ ��������� ������� ������ .  
���� �������� ![](http://latex.codecogs.com/gif.latex?%5Cinline%20P_y)  � ![](http://latex.codecogs.com/gif.latex?%5Cinline%20p_%7By%7D%28x%29), �� ������� �������� ����� 
![](http://latex.codecogs.com/gif.latex?%5Cinline%20R%28a%29%20%3D%20%5Csum_%7By%5Cepsilon%20Y%7D%20%5Csum_%7Bs%5Cepsilon%20Y%7D%20%5Clambda_yP_yP%28A_s%7Cy%29)
![](http://latex.codecogs.com/gif.latex?%5Cinline%20A_s%20%3D%20%5Cbigl%5C%7Bx%20%5Cin%20X%7Ca%28x%29%3Ds%5Cbigr%5C%7D%2C)  
����������� ���������� ![](http://latex.codecogs.com/gif.latex?%5Cinline%20a%28x%29%20%3D%20%5Carg%5Cmax%20%5Clambda_yP_yp_y%28x%29)

2) ***�������������� ���������� �� �������***  
�� ����������  ������ *y* ������ ������������ ������  ![](http://latex.codecogs.com/gif.latex?%5Cinline%20P_y) (���� �������� � �������) � ![](http://latex.codecogs.com/gif.latex?%5Cinline%20p_y%28x%29).  
��� ������:  
- **���������������** ���� ��������� ���������� (�����������) - ��� � ���;  
- **�����������������** - ������ ������� - �����������, ����� ������������� ����;   
- **H��������� �����** ������������ _��-����������_. ��������� ��������� ����� (����������� ���������) - ���������� �������,����� ���������� �������� �������.

## Plugin
__�������������� ��������__ � �������� ������� ����������������� ����������
������������� ����������� ���������� ���������, ������� ����������� �� �������:

![](http://latex.codecogs.com/svg.latex?N%28x%3B%5Cmu%2C%5CSigma%29%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B%282%5Cpi%29%5En%20%7C%5CSigma%7C%7D%7D%20%5Ccdot%20exp%5Cleft%28-%5Cfrac%7B1%7D%7B2%7D%28x%20-%20%5Cmu%29%5ET%20%5CSigma%5E%7B-1%7D%20%28x%20-%20%5Cmu%29%5Cright%29),
� �������

![](http://latex.codecogs.com/svg.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5En)
� ������, ��������� �� *n* ���������,

![](http://latex.codecogs.com/svg.latex?%5Cmu%20%5Cin%20%5Cmathbb%7BR%7D%5En)
� �������������� ��������,

![](http://latex.codecogs.com/svg.latex?%5CSigma%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D)
� �������������� �������.

����� ������ _��������� ������������� �������_, �������� ���������������
����������� ��������� ![](http://latex.codecogs.com/svg.latex?%5Cmu%2C%20%5CSigma) �� ��������� �������� ��� ������� ������ _y_ :

![](http://latex.codecogs.com/svg.latex?%5Chat%7B%5Cmu%7D%20%3D%20%5Cfrac%7B1%7D%7Bl_y%7D%20%24%24%5Csum_%7Bi%20%3D%201%7D%5E%7Bl_y%7D%20x_i%24%24)

![](http://latex.codecogs.com/svg.latex?%5Chat%7B%5CSigma%7D%20%3D%20%5Cfrac%7B1%7D%7Bl_y%20-%201%7D%20%24%24%5Csum_%7Bi%20%3D%201%7D%5E%7Bl_y%7D%20%28x_i%20-%20%5Chat%7B%5Cmu%7D%29%28x_i%20-%20%5Chat%7B%5Cmu%7D%29%5ET).



```R
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
```
**����������**

![bc](https://raw.githubusercontent.com/bestful/ML/master/samples/plugin.png)

�� ����� ������������� �����, ��� ����������� ����������� �������� ������������.
����� ���� ����������, LOO � ���� ����� ����������� - 0.02.


### ����������

 - ������� ������������� ������� ����� ���������� �� �����������: ����� ������� ���������� ��������, �����������  ��������, ���
����� ������ ����������� �� ������������� �������.
 - ���� ����� ������� ������ ����������� ������������ ��� ����� ��������� ���� ������� ���������, �� �������������� ������� ���������� �����������. � ���� ������ �������� ������� �� ���������� � ����� ������ ����������.
 - ���������� ������ ������������� � ���������� ������������ �������������, � ���������, � ������ ������� ��������.



## LDF 

***�������������� �������*** ������� �����, ������ **s, t** ������������� � ����������� ![](https://latex.codecogs.com/gif.latex?%5Clambda_sP_s%20%3D%20%5Clambda_tP_t), �������� ��������������� � ����� ���������� ***���������*** .  

��� ��������, ��� ������ ����� ���������� ����������� �����, ����������� ��������� �������� ���������� ����� ��������, ������������ �����, ����������� ������ �������. ������� ���������� - ������, � ���������� �������� �� ������� ������ ����������� ��������� �������,� ���������� ����������� ������ **R(a)**.

��� �������� �������� � ��������� ����� ����� �������������� ������ �������� ������������� ��������� ���: 
![](http://www.machinelearning.ru/mimetex/?a(x)%20=%20\mathrm{arg}\max_{y\in%20Y}%20(\ln(\lambda_{y}%20P_y)%20-%20\frac{1}{2}\mu_{y}^{T}%20\Sigma^{-1}%20\mu_y%20+%20x^T%20\Sigma^{-1}%20\mu_y))

����������
``` R
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
```
**����������**

![bc](https://raw.githubusercontent.com/bestful/ML/master/samples/fisher.png)

�������� ������� ��������, ����� ����� ������� ������������� ������ � ���������� � �� ������� ������ �����������.  

� ���� ������ �������� �������� ������� ������ � ������������ ������������, �� ���������� �������������, � ����� �������� ������ ���������� ������������.
## �������� ��������������
**�������� �������������** � �������� �������������, ���������� �� ���������� �������� ����������� �����������. � ������ ���� ������� ����������� ������������ �������� ��������������, ������� ����� ������������ ��������� �� ��� ����������������. � ������ �������� ����� ������� ����������� ����������� �������-�������. 

���������� ������ 2 �������.
����� ![](http://www.machinelearning.ru/mimetex/?Y=\\{-1,+1\\})

�������� ��������������� ���������� �������� ������������� ![](http://www.machinelearning.ru/mimetex/?a:%20X\to%20Y) ����
![](https://latex.codecogs.com/svg.latex?sign%28%3Cw%2C%20x%3E%29)
��� ![](http://www.machinelearning.ru/mimetex/?w_j) � ��� j-�� ��������, ![](http://www.machinelearning.ru/mimetex/?w_0) � ����� �������� �������, w � ������ �����. ��������������, ��� ������������ ����� ������������ ������� �������: ![](http://www.machinelearning.ru/mimetex/?f_{0}(x)=-1). 

�������� ![](http://latex.codecogs.com/svg.latex?M_i%28w%29%3Dy_i%5Clangle%20x_i%2Cw%20%5Crangle)���������� __��������__ ������� ������������ ��������� �������������. ���� ![](http://latex.codecogs.com/svg.latex?M_i%28w%29%3C0),�������� ��������� �� ������� ![](http://latex.codecogs.com/svg.latex?x_i) ������.

![](http://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D%28M%29)� ��������� �������������� __������� ������__, ���������� ��������� �������
![](http://latex.codecogs.com/svg.latex?%5BM%3C0%5D%20%5Cleq%20%5Cmathcal%7BL%7D%28M%29).

����� __���������� ��������� ������__ ����� ������������� ��� ������� ����
![](https://latex.codecogs.com/svg.latex?%5Ctilde%7BQ%7D%28w%2CX%5E%5Cell%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7B%5Cell%7DL%5Cmathcal%28M_i%28w%29%29%5Crightarrow%20%5Cmin_w)

### ����� ��������������� ���������
����� ������ ��������� ������� ![](https://latex.codecogs.com/gif.latex?X%5El%20%3D%20%5C%7B%28x_i%2C%20y_i%29%5C%7D_%7Bi%3D1%7D%5El%20%2C%20x_i%20%5Cin%20%5Cmathbb%7BR%7D%5En%20%2C%20y_i%20%5Cin%20%5C%7B-1%2C%20&plus;1%20%5C%7D)

��������� ����� ������ ���������� ![](https://latex.codecogs.com/gif.latex?w%20%5Cin%20%5Cmathbb%7BR%7D) , ��� ������� ����������� ������� ������������������� ������������� �����:

![](https://latex.codecogs.com/gif.latex?Q%28w%2C%20X%5El%29%20%3D%20%5Csum_%7Bi%3D1%7D%5El%20L%28%5Clangle%20w%2C%20x_i%20%5Crangle%20y_i%29%20%5Crightarrow%20%5Cmin_w)

�������� ��� ����������� Q(w) ����� ������������ ������. � ���� ������ ���������� ��������� ��������� ����������� ��� w, ����� ����������� ������������ �������, �� ������ ���� �������� ������ w ���������� � ����������� �������� �������� �������� ����������� Q. ��� ����������� �������������� ����������� ������� ��������� ![](https://latex.codecogs.com/gif.latex?Q%27%28w%29%20%3D%20%5Cleft%28%5Cdfrac%7B%5Cpartial%20Q%28w%29%7D%7B%5Cpartial%20w_j%7D%5Cright%29_%7Bj%3D1%7D%5En) :

![](https://latex.codecogs.com/gif.latex?w%20%3D%20w%20-%20%5Ceta%20Q%27%28w%29)
��� ![](https://latex.codecogs.com/gif.latex?%5Ceta%20%3E%200) - ���� ��������

�����������, ��� ������� L ���������������. ������� ��������:

![](https://latex.codecogs.com/gif.latex?w%20%3D%20w%20-%20%5Ceta%20%5Csum_%7Bi%3D1%7D%5En%20L%27%28%5Clangle%20w%2C%20x_i%20%5Crangle%20y_i%29%20x_i%20y_i)

������ ��������� ![](https://latex.codecogs.com/gif.latex?%28x_i%2C%20y_i%29) ������ ���������� ����� � ��������� ������� w, �� ������ w ���������� ������ ����� �������� ���� l ��������. ���������� ������������� �������� ����� ��������, ���� �������� ���������� ![](https://latex.codecogs.com/gif.latex?%28x_i%2C%20y_i%29) �� ������, ��� ������� ������ ����������� ��� � ����� ��������� ������ �����:
![](https://latex.codecogs.com/gif.latex?w%20%3D%20w%20-%20%5Ceta%20L%27_a%20%28%5Clangle%20w%2C%20x_i%20%5Crangle%20y_i%29x_i%20y_i)

� ������ __��������������� ��������� (stochastic gradient, SG)__ ���������� ������������ � ��������� �������. ���� �� ������� ����������� � ��������� ������������� �������, ������� ����� ����������� ��� ���������.������������� ����� ����� ������������� ���������� ���������. ����������� ������������ - ����� ��������� ��������� �������� ![](https://latex.codecogs.com/gif.latex?w_j%20%3D%20random%5Cleft%28%5Cfrac%7B-1%7D%7B2n%7D%3B%20%5Cfrac%7B1%7D%7B2n%7D%5Cright%29)

### ������������
- ����� ����� ����������� � ����� ���������� �� ���������� �������������� � �� ��������� ���� ? ������������ �������� ���������������.
- ����� �������� ��� ������������� ��������, ����� ��������� ������� ��������� �������, � ������ ����� ����������� ��� ��������������� �������.
- ����� ��������� ����������� ���� �� ��������� ������� ��������, �� ���� ����, ��� ��������� ���������� ����� ��������� ���������� ��� ��������.

### ����������
- ���������� Q, ��� �������, ������������������, � ������� ����� ��������� � ���������� ��������, ��������� ����� �������� ��� �� ��������� �����.
- ��� ������� ����������� ������������ n ��� ����� ����� ������� l �������� ������������. ��� ���� ����� ���������� ����� ������� �����, ���������� ������� �� ���������� �������� ������������� � ������������� ����, ������������� ���������� ������������ -- ����� ��������� ��������� �������, ���������� �����������, ������� ������������ �������� ��� ���������� ��������� ����� ������ �������� �������������� ������ �����, ������������� ����������� ��������� ������������� ����� ��������.
- ���� ������� ������ ����� �������������� ���������, �� ������� ����� ������� � ��������� ?���������. ��� ������ �������� ���������� ������������  , ��� ����� �������� ����������� L' � ����, ��� ������ ���������� �����. ���� ���� w ������ � ������� ������� ��������, �� � ��� ����������� �� �������� ������ ��������� �� ���� ������� �����.

����������:
``` R
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
    Q.prev <- Q
    Q <- (1 - lam) * Q + lam * err
    callback(w, x)
    it <- it+1
  }
  
  w
}
```

## ���������� �������� �������
����� _������������ ������� ������_
![](http://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D%28M%29%3D%28M-1%29%5E2%3D%28%5Clangle%20w%2Cx_i%20%5Crangle%20y_i-1%29%5E2)

� _������-�������_ ������� ���������� �����
![](http://latex.codecogs.com/svg.latex?w%3Dw-%5Ceta%28%5Clangle%20w%2Cx_i%20%5Crangle-y_i%29x_i).

���������� ��������:
``` R
learn.lin.adaline <- function(xl, ...){
  lin.SG(xl, 
         function(M){ 
           (M-1)**2 
         }, 
         function(w, temp, xi, yi){
           w - temp*(sum(w*xi)-yi)*xi
         }, ...)
}
```
������ �� ������ ������ � 2 �������� (������ virginica). 
![lic](https://raw.githubusercontent.com/bestful/ML/master/samples/adaline.png)
����� ������ ���������� ����������� ����������� �� ��������� ��������

## ���������� �����������
����� _�������-�������� ������� ������_
![](http://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D%3D%28-M%29_&plus;%3D%5Cmax%28-M%2C0%29)

� _������� �����_ ��� ���������� �����
![](http://latex.codecogs.com/svg.latex?%5Ctext%7Bif%20%7D%5Clangle%20w%2Cx_i%20%5Crangle%20y_i%3C0%20%5Ctext%7B%20then%20%7D%20w%3A%3Dw&plus;%5Ceta%20x_iy_i).

���������� ��������:
``` R
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
```
![lic](https://raw.githubusercontent.com/bestful/ML/master/samples/perceptron.png)
### ������������� ���������

����� �������� __����������� ����������� ���������������__ ��-�� ����� ��������
������� ������������� �������������.

����� _������������� ������� ������_
![](http://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D%28M%29%20%3D%20%5Clog_2%281%20&plus;%20e%5E%7B-M%7D%29)

� _�������������_ ������� ���������� �����
![](http://latex.codecogs.com/svg.latex?w%20%3A%3D%20w&plus;%5Ceta%20y_ix_i%5Csigma%28-%5Clangle%20w%2Cx_i%20%5Crangle%20y_i%29) , ���
![](http://latex.codecogs.com/svg.latex?%5Csigma%28z%29%3D%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-z%7D%7D) � _���������� �������_.
���������� ��������:
``` R
learn.lin.logistic <- function(xl, ...){
  lin.SG(xl, 
              function(M){ 
                log2(1+exp(-M))
              }, 
              function(w, temp, xi, yi){
                w + temp*yi*xi*(1/(1+exp(sum(w*xi)*yi)))
              }, ...)
}
```

![lic](https://raw.githubusercontent.com/bestful/ML/master/samples/logistic.png)
### �������, ���������� ����������� � ������������� ���������
![lic](https://raw.githubusercontent.com/bestful/ML/master/samples/lin_all.png)
## SVM 
����� ������� ������� ��������� � ���������� �������� ���������� ![](https://latex.codecogs.com/gif.latex?w_0) � ![](https://latex.codecogs.com/gif.latex?w_0) ��� ������� ���������� ����� ������

![](https://latex.codecogs.com/gif.latex?Q%28w%2C%20w_0%29%20%3D%20%5Csum_%7Bi%3D1%7D%5El%20%5By_i%28%5Clangle%20w%2C%20x_i%5Crangle%20-%20w_0%29%20%5Cleq%200%5D)

��������� ������� ��������. �� ����� ����������� �������������� �� �����������. ����� �������� �� ����� �������, ����� ��� ������� __����������� ������ �� ��������� � ��� ����� ����� �������__. 

�������, ��� �������� a(x) �� ���������, ���� ![](https://latex.codecogs.com/gif.latex?w_0) � ![](https://latex.codecogs.com/gif.latex?w_0) ������������ �������� �� ���� � �� �� ������������� ���������. ������� ��� ��������� ���, ����� ����������� �������

![](https://latex.codecogs.com/gif.latex?%5Cmin_%7Bi%3D1%2C%20%5Cdots%2C%20l%7D%20y_i%20%28%5Clangle%20w%2C%20x_i%20%5Crangle%20-%20w_0%29%20%3D%201)

��������� ����� ![](https://latex.codecogs.com/gif.latex?%5C%7B%20x%3A%20-1%20%5Cleq%20%5Clangle%20w%2C%20x%20%5Crangle%20-w_0%20%5Cleq%201%20%5C%7D) ��������� ������, ����������� ������.
![](https://raw.githubusercontent.com/bestful/ML/master/readme/SVM_margins.png)

����������� �������������� �������� ����� �� �������� ����� ����. �������, ��������� � ����������� ��������������, ����� �� �������� ������, � ������ �� ��� ����������� �������. � ������ �� ������� ������� ���� �� ���� ����� ������.
����, � ������ ������� ���������� ������� �������� ������ ������������� ����������������:

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bcases%7D%20%5Clangle%20w%2C%20w%20%5Crangle%20%5Crightarrow%20min%20%5C%5C%20y_i%28%5Clangle%20w%2C%20x_i%20%5Crangle%20-%20w_0%29%20%5Cgeq%201%2C%20i%20%3D%201%2C%20%5Cdots%2C%20l%20%5Cend%7Bcases%7D)

������� ���������� ������ �� ������ ������� ������������ �������: �������� ��������� ��������� ������ �� ��������� ��������, �� ���, ����� ������ ���� ��� ����� ������. 

��� ����� ������ �������������� ���������� ![](https://latex.codecogs.com/gif.latex?%5Cepsilon_i%20%5Cgeq%200), ��������������� �������� ������ �� �������� �������� ![](https://latex.codecogs.com/gif.latex?x_i) ����� ������ ���������� ������ ����� ����� ������

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bcases%7D%201/2%20%5Clangle%20w%2C%20w%20%5Crangle%20&plus;%20C%20%5Csum_%7Bi%3D1%7D%5El%20%5Cepsilon_i%20%5Crightarrow%20%5Cmin_%7Bw%2C%20w_0%2C%20%5Cepsilon%7D%20%5C%5C%20y_i%20%28%5Clangle%20w%2C%20x_i%5Crangle%20-%20w_0%29%20%5Cgeq%201%20-%20%5Cepsilon_i%20%5Cend%7Bcases%7D)

������������� ��������� C �������� ����������� ���������� ������ � ��������� �������� ���������� ����� ������������� ������ ����������� ������ � ������������ ��������� ������.

___������ SVM___
``` R
require("e1071")
require("kernlab")

ssel <- sel[ Species!="virginica" ,]
ssel$Species <- factor(ssel$Species)
smodel <- ksvm(Species ~ Petal.Length + Petal.Width, data = ssel, kernel="linea")
plot(smodel, ssel, Petal.Width ~ Petal.Length)

ssel <- sel[ Species!="setosa" ,]
ssel$Species <- factor(ssel$Species)
smodel <- svm(Species ~ Petal.Length + Petal.Width, data = ssel, kernel="linea")
plot(smodel, ssel, Petal.Width ~ Petal.Length, bgcol="red")
```

![](https://raw.githubusercontent.com/bestful/ML/master/samples/svm_1.png)
![](https://raw.githubusercontent.com/bestful/ML/master/samples/svm_2.png)
