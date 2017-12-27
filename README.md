# ML
# Метрические алгоритмы классификации
**Метрические методы обучения** -- методы, основанные на анализе сходства объектов.

**_Мерой близости_** называют функцию расстояния ![](http://latex.codecogs.com/svg.latex?%5Clarge%20%5Crho%3A%20%28X%20%5Ctimes%20X%29%20%5Crightarrow%20%5Cmathbb%7BR%7D). Чем меньше расстояние между объектами, тем больше объекты похожи друг на друга.

Метрические алгоритмы классификации опираются на **_гипотезу компактности_**: схожим объектам соответствуют схожие ответы.

Метрические алгоритмы классификации с обучающей выборкой *Xl* относят объект *u* к тому классу *y*, для которого **суммарный вес ближайших обучающих объектов ![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29) максимален**:

![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29%20%3D%20%5Csum_%7Bi%20%3A%20y_%7Bu%7D%5E%7B%28i%29%7D%20%3D%20y%7D%20w%28i%2C%20u%29%20%5Crightarrow%20max)

, где весовая функция *w(i, u)* оценивает степень важности *i*-го соседа для классификации объекта *u*.

Функция ![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29) называется **_оценкой близости объекта u к классу y_**. Выбирая различную весовую функцию *w(i, u)* можно получать различные метрические классификаторы.

Для поиска оптимальных параметров для каждого из рассматриваемых ниже метрических алгоритмов используется **LOO -- leave-one-out** *(критерий скользящего контроля)*, который состоит в следующем: 

1. Исключать объекты *x(i)* из выборки *Xl* по одному, получится новая выборка без объекта *x(i)* (назовём её *Xl_1*).
2. Запускать алгоритм от объекта *u*, который нужно классифицировать, на выборке *Xl_1*.
3. Завести переменную *Q* (накопитель ошибки, изначально *Q = 0*) и, когда алгоритм ошибается, *Q = Q + 1*.
4. Когда все объекты *x(i)* будут перебраны, вычислить *LOO = Q / l* (*l* -- количество объектов выборки).

При минимальном значении LOO получим оптимальный параметр алгоритма.

## Алгоритм k ближайших соседей (knn)


Для оценки близости объекта _u_ к классу _y_ алгоритм использует следующую
функцию:
![](http://latex.codecogs.com/svg.latex?%5Clarge%20W%28i%2C%20u%29%20%3D%20%5Bi%20%5Cleq%20k%5D)
, где _i_ обозначает порядок соседа по расстоянию к точке _u_.


Алгоритм выбирает _k_ ближайших соседей и возвращает
тот класс, который среди выбранных встречается большее количество раз.

Программно алгоритм реализуется следующим образом:
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
При k=1 получаем LOO = 0.047
А теперь посмотрим LOO при различных k
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_knn.png)
Наилучший результат получаем при k=6
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/6nn.png)
Посмотрим на преимущества и недостатки

### Преимущества:
1. Простота реализации.
2. При *k*, подобранном около оптимального, алгоритм "неплохо" классифицирует.

### Недостатки:
1. Нужно хранить всю выборку.
2. При *k = 1* неустойчивость к погрешностям (*выбросам* -- объектам, которые окружены объектами чужого класса), вследствие чего этот выброс классифицировался неверно и окружающие его объекты, для которого он окажется ближайшим, тоже.
2. При *k = l* алгоритм наоборот чрезмерно устойчив и вырождается в константу.
3. Максимальная сумма объектов в *counts* может достигаться в нескольких классах одновременно.
4. "Скудный" набор параметров.
5. Точки, расстояние между которыми одинаково, не все будут учитываться.

## Алгоритм k взвешеных ближайших соседей (wknn)
Имеется некоторая выборка *Xl*, состоящая из объектов *x(i), i = 1, ..., l* (в приложенной программе используется выборка ирисов Фишера).
Данный алгоритм классификации относит объект *u* к тому классу *y*, у которого максимальна сумма весов *w_i* его ближайших *k* соседей *x(u_i)*.

Для оценки близости классифицируемого объекта *u* к классу *y* **алгоритм wknn** использует следующую функцию:

![](http://latex.codecogs.com/svg.latex?%5Clarge%20W%28i%2C%20u%29%20%3D%20%5Bi%20%5Cleq%20k%5D%20w%28i%29) , где *i* -- порядок соседа по расстоянию к классифицируемому объекту *u*, а *w(i)* -- строго убывающая функция веса, задаёт вклад i-го соседа в классификацию.

В приложенной программе используется весовая функция вида: ![](https://latex.codecogs.com/gif.latex?w%28i%29%20%3D%20q%5Ei%2C%20q%20%5Cepsilon%20%280%2C%201%29)

Реализация весовой функции:

``` R
mc.wlin <- function(el){
  k <- el[1]
  i <- el[2]
  (k+1-i)/k
}

```

Реализация классификатора:
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
Давайте найдем оптимальный k в kwnn через LOO
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_kwnn.png)
Получаем то же оптимальное k. 
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/6wnn.png)

![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/parzen.png)
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_parzen.png)
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_parzen_auto.png)
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/poten.png)
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/stolp.png)