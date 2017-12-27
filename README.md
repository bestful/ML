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
1. �������� ����������.
2. ��� *k*, ����������� ����� ������������, �������� "�������" ��������������.

### ����������:
1. ����� ������� ��� �������.
2. ��� *k = 1* �������������� � ������������ (*��������* -- ��������, ������� �������� ��������� ������ ������), ���������� ���� ���� ������ ����������������� ������� � ���������� ��� �������, ��� �������� �� �������� ���������, ����.
2. ��� *k = l* �������� �������� ��������� �������� � ����������� � ���������.
3. ������������ ����� �������� � *counts* ����� ����������� � ���������� ������� ������������.
4. "�������" ����� ����������.
5. �����, ���������� ����� �������� ���������, �� ��� ����� �����������.

## �������� k ��������� ��������� ������� (wknn)
������� ��������� ������� *Xl*, ��������� �� �������� *x(i), i = 1, ..., l* (� ����������� ��������� ������������ ������� ������ ������).
������ �������� ������������� ������� ������ *u* � ���� ������ *y*, � �������� ����������� ����� ����� *w_i* ��� ��������� *k* ������� *x(u_i)*.

��� ������ �������� ����������������� ������� *u* � ������ *y* **�������� wknn** ���������� ��������� �������:

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
�������� �� �� ����������� k. 
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/6wnn.png)

![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/parzen.png)
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_parzen.png)
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_parzen_auto.png)
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/poten.png)
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/stolp.png)