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
- Простота реализации.
- При *k*, подобранном около оптимального, алгоритм "неплохо" классифицирует.

### Недостатки:
- Нужно хранить всю выборку.
- При *k = 1* неустойчивость к погрешностям (*выбросам* -- объектам, которые окружены объектами чужого класса), вследствие чего этот выброс классифицировался неверно и окружающие его объекты, для которого он окажется ближайшим, тоже.
- При *k = l* алгоритм наоборот чрезмерно устойчив и вырождается в константу.
- Максимальная сумма объектов в *counts* может достигаться в нескольких классах одновременно.
- "Скудный" набор параметров.
- Точки, расстояние между которыми одинаково, не все будут учитываться.

## Алгоритм k взвешеных ближайших соседей (kwnn)
Имеется некоторая выборка *Xl*, состоящая из объектов *x(i), i = 1, ..., l* (в приложенной программе используется выборка ирисов Фишера).
Данный алгоритм классификации относит объект *u* к тому классу *y*, у которого максимальна сумма весов *w_i* его ближайших *k* соседей *x(u_i)*.

Для оценки близости классифицируемого объекта *u* к классу *y* **алгоритм kwnn** использует следующую функцию:

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
Получаем диапазон оптимальных k больше. 
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/6wnn.png)
Чем kwnn лучше/хуже knn?
- То же самое, что и knn
- Шире диапазон оптимальных k.
- Лучше точность на границах.

## Метод парзеновского окна
Имеется некоторая выборка *Xl*, состоящая из объектов *x(i), i = 1, ..., l* (в приложенной программе используется выборка ирисов Фишера). В данном алгоритме весовая функция *w_i* определяется как функция **от расстояния между классифицируемым объектом *u* и его соседями *x(u_i), i = 1, ..., l*, а не от ранга соседа *i***, как было в весовом kNN.

Для оценки близости классифицируемого объекта *u* к классу *y* **метод парзеновского окна** использует следующую функцию:

![](https://latex.codecogs.com/gif.latex?W%28i%2C%20u%29%20%3D%20K%5Cleft%20%28%20%5Cfrac%7B%5Crho%20%28u%2C%20x_%7Bu%7D%5E%7Bi%7D%29%7D%7Bh%7D%20%5Cright%20%29) , где *K(z)* -- функция ядра (не возрастающая от 0 до бесконечности), а *h* -- ширина окна (окно -- сферическая окрестность классифицируемого объекта *u* радиуса *h*).

Рассматриваются 5 ядер:
- Прямоугольное 

![](https://latex.codecogs.com/gif.latex?K%28z%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5B%7Cz%7C%20%3C%3D%201%5D)
- Треугольное 

![](https://latex.codecogs.com/gif.latex?K%28z%29%20%3D%20%281%20-%20%7Cz%7C%29%5B%7Cz%7C%20%3C%3D%201%5D)
- Квартическое 

![](https://latex.codecogs.com/gif.latex?K%28z%29%20%3D%20%5Cfrac%7B15%7D%7B16%7D*%281%20-%20z%5E%7B2%7D%29%5B%7Cz%7C%20%3C%3D%201%5D)
- Гауссовское 

![](https://latex.codecogs.com/gif.latex?K%28z%29%20%3D%20%28%282%5Cpi%29%5E%7B%5E%7B%28%5Cfrac%7B-1%7D%7B2%7D%7D%29%7D%29*e%5E%7B%28%5Cfrac%7B-z%5E2%7D%7B2%7D%29%7D)
- Епанечникова 


![](https://latex.codecogs.com/gif.latex?K%28z%29%20%3D%20%5Cfrac%7B3%7D%7B4%7D*%281%20-%20z%5E2%29%5B%7Cz%7C%20%3C%3D%201%5D)

Реализация классификатора:
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
Находим оптимальный шаг:
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_parzen.png)
Особой разницы между ядрами в данной выборке нет. Возьмем треугольное ядро с оптимальным h = 0.4.
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/parzen.png)
Посмотрим на достоинства и недостатки:

### Плюсы
- хорошее качество классификации при правильно подобраном _h_
- все точки с одинаковым расстоянием будут учитаны

### Минусы
- необходимо хранить всю выборку целиком
- диапазон параметра _h_ необходимо подбирать самостоятельно, учитывая
плотность расположения точек
- если ни одна точка не попала в радиус _h_, алгоритм не способен ее
классифицировать (не актуально для гауссовского ядра)

## Метод парзеновского окна с переменным окном
То же самое, что и прошлый алгоритм, только каждый раз шаг подбирается в зависимости от расстояния до k соседа. 
Посмотрим LOO для различных ядер:
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/loo_parzen_auto.png)
Прямоугольное и Гауссовское ядро в начальных k дают результат лучше, чем с постоянным шагом.
В целом ничего удивительного, получаем:
- прямоугольное ядро -> knn
- треугольное ядро -> kwnn c линейным w[i]
- гауссовское ядро -> kwnn cо специальным w[i]

## Метод потенциальных функций

Имеется некоторая выборка *Xl*, состоящая из объектов *x(i), i = 1, ..., l* (в приложенной программе используется выборка ирисов Фишера). В данном алгоритме весовая функция *w_i* определяется как функция от расстояния между классифицируемым объектом *u* и его соседями *x(u_i), i = 1, ..., l*, как и в **методе парзеновского окна**.

Для оценки близости классифицируемого объекта *u* к классу *y* **метод потенциальных функций** использует следующую функцию:

![](https://latex.codecogs.com/gif.latex?W%28i%2C%20u%29%20%3D%20%5Cgamma_%7Bi%7D*K%5Cleft%20%28%20%5Cfrac%7B%5Crho%20%28u%2C%20x_%7Bu%7D%5E%7Bi%7D%29%7D%7Bh_i%7D%20%5Cright%20%29%2C%20%5Cgamma_%7Bi%7D%20%5Cgeqslant%200%2C%20h_i%20%3E%200) 

**Основная идея:** *Потенциалы* определяют важность каждого объекта *x_i* при классификации. Считаем, что радиусы потенциалов *h* известны заранее. Алгоритм подбирает только потенциалы ![](https://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D). 

Построение потенциалов:
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

Реализация алгоритма:
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

**Важное отличие алгоритма потенциальных функций от предыдущих алгоритмов:** центр "окна" располагается в обучающем объекте, а не в классифицируемом.

Карта классификации:
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/poten.png)

### Плюсы:
- результат зависит от _2l_ параметров

### Минусы:
- необходимо хранить всю выборку целиком
- параметры _h_ необходимо подбирать самостоятельно, алгоритм
в их подборе не принимает участия
- если ни одна точка не попала в радиус _h_, алгоритм не способен ее
классифицировать (не актуально для гауссовского ядра)
- медленно сходится
- слишком грубо настраивает параметры
- неопределенное время работы (при маленьком пороге ошибки может вообще
выполняться бесконечно)

## STOLP
В разработке...
![mc](https://raw.githubusercontent.com/bestful/ML/master/samples/stolp.png)

# Байесовские алгоритмы классификации
Байесовские алгоритмы классификации основаны на принципе максимума апостериорной вероятности. Для классифицируемого объекта вычисляются плотности распределения 
- **_функции правдоподобия_** классов ![](http://latex.codecogs.com/gif.latex?%5Cinline%20p%28x%7Cy%29%20%3D%20p_y%28x%29)
-  ***априорные вероятности*** классов ![](http://latex.codecogs.com/gif.latex?%5Cinline%20P_y)

По ним вычисляются ***апостериорные вероятности*** - ![](http://latex.codecogs.com/gif.latex?p%20%5Cleft%20%5C%7By%7Cx%20%5Cright%20%5C%7D%20%3D%20P_yp_y%28x%29)
Объект относится к классу с максимальной апостериорной вероятностью.

*Задача классификации* - получить алгоритм ![](http://latex.codecogs.com/gif.latex?%5Cinline%20a%3A%5C%3B%20X%5Cto%20Y), способный классифицировать произвольный объект ![](http://latex.codecogs.com/gif.latex?%5Cinline%20x%20%5Cin%20X).  

1)  ***Построение классификатора при известных плотностях***  
![](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Clambda_y) - штраф за неправильное отнесение объекта класса .  
Если известны ![](http://latex.codecogs.com/gif.latex?%5Cinline%20P_y)  и ![](http://latex.codecogs.com/gif.latex?%5Cinline%20p_%7By%7D%28x%29), то минимум среднего риска 
![](http://latex.codecogs.com/gif.latex?%5Cinline%20R%28a%29%20%3D%20%5Csum_%7By%5Cepsilon%20Y%7D%20%5Csum_%7Bs%5Cepsilon%20Y%7D%20%5Clambda_yP_yP%28A_s%7Cy%29)
![](http://latex.codecogs.com/gif.latex?%5Cinline%20A_s%20%3D%20%5Cbigl%5C%7Bx%20%5Cin%20X%7Ca%28x%29%3Ds%5Cbigr%5C%7D%2C)  
достигается алгоритмом ![](http://latex.codecogs.com/gif.latex?%5Cinline%20a%28x%29%20%3D%20%5Carg%5Cmax%20%5Clambda_yP_yp_y%28x%29)

2) ***Восстановление плотностей по выборке***  
По подвыборке  класса *y* строим эмпирические оценки  ![](http://latex.codecogs.com/gif.latex?%5Cinline%20P_y) (доля объектов в выборке) и ![](http://latex.codecogs.com/gif.latex?%5Cinline%20p_y%28x%29).  
Три метода:  
- **Параметрический** если плотности нормальные (гауссовские) - НДА и ЛДФ;  
- **Непараметрический** - оценка Парзена - Розенблатта, метод парзеновского окна;   
- **Hазделение смеси** производится _ЕМ-алгоритмом_. Плотности компонент смеси (гауссовские плотности) - радиальные функции,метод радиальных базисных функций.

## Plugin
__Подстановочный алгоритм__ в качестве моделей восстанавливаемых плотностей
рассматривает многомерные нормальные плотности, которые вычисляются по формуле:

![](http://latex.codecogs.com/svg.latex?N%28x%3B%5Cmu%2C%5CSigma%29%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B%282%5Cpi%29%5En%20%7C%5CSigma%7C%7D%7D%20%5Ccdot%20exp%5Cleft%28-%5Cfrac%7B1%7D%7B2%7D%28x%20-%20%5Cmu%29%5ET%20%5CSigma%5E%7B-1%7D%20%28x%20-%20%5Cmu%29%5Cright%29),
в которой

![](http://latex.codecogs.com/svg.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5En)
– объект, состоящий из *n* признаков,

![](http://latex.codecogs.com/svg.latex?%5Cmu%20%5Cin%20%5Cmathbb%7BR%7D%5En)
– математическое ожидание,

![](http://latex.codecogs.com/svg.latex?%5CSigma%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D)
– ковариационная матрица.

Чтобы узнать _плотности распределения классов_, алогритм восстанавливает
неизвестные параметры ![](http://latex.codecogs.com/svg.latex?%5Cmu%2C%20%5CSigma) по следующим формулам для каждого класса _y_ :

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
**Результаты**

![bc](https://raw.githubusercontent.com/bestful/ML/master/samples/plugin.png)

На карте классификации видно, что разделяющая поверхность является квадратичной.
Среди всех алгоритмов, LOO у него самый минимальный - 0.02.


### Недостатки

 - Функции правдоподобия классов могут отличаться от гауссовских: когда имеются дискретные признаки, принимающие  значения, или
когда классы распадаются на изолированные участки.
 - Если длина выборки меньше размерности пространства или среди признаков есть линейно зависимые, то ковариационная матрица становится вырожденной. В этом случае обратная матрица не существует и метод вообще неприменим.
 - Выборочные оценки чувствительны к нарушениям нормальности распределений, в частности, к редким большим выбросам.



## LDF 

***Ковариационные матрицы*** классов равны, классы **s, t** равновероятны и равнозначны ![](https://latex.codecogs.com/gif.latex?%5Clambda_sP_s%20%3D%20%5Clambda_tP_t), признаки некоррелированы и имеют одинаковые ***дисперсии*** .  

Это означает, что классы имеют одинаковую сферическую форму, разделяющая плоскость проходит посередине между классами, ортогонально линии, соединяющей центры классов. Нормаль оптимальна - прямая, в одномерной проекции на которую классы разделяются наилучшим образом,с наименьшим байесовским риском **R(a)**.

При принятии гипотезы о равенстве между собой ковариационных матриц алгоритм классификации принимает вид: 
![](http://www.machinelearning.ru/mimetex/?a(x)%20=%20\mathrm{arg}\max_{y\in%20Y}%20(\ln(\lambda_{y}%20P_y)%20-%20\frac{1}{2}\mu_{y}^{T}%20\Sigma^{-1}%20\mu_y%20+%20x^T%20\Sigma^{-1}%20\mu_y))

Реализация
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
**Результаты**

![bc](https://raw.githubusercontent.com/bestful/ML/master/samples/fisher.png)

Алгоритм неплохо работает, когда формы классов действительно близки к нормальным и не слишком сильно различаются.  

В этом случае линейное решающее правило близко к оптимальному байесовскому, но устойчивее квадратичного, и часто обладает лучшей обобщающей способностью.
## Линейные классификаторы
**Линейный классификатор** — алгоритм классификации, основанный на построении линейной разделяющей поверхности. В случае двух классов разделяющей поверхностью является гиперплоскость, которая делит пространство признаков на два полупространства. В случае большего числа классов разделяющая поверхность кусочно-линейна. 

Рассмотрим случай 2 классов.
Пусть ![](http://www.machinelearning.ru/mimetex/?Y=\\{-1,+1\\})

Линейным классификатором называется алгоритм классификации ![](http://www.machinelearning.ru/mimetex/?a:%20X\to%20Y) вида
![](https://latex.codecogs.com/svg.latex?sign%28%3Cw%2C%20x%3E%29)
где ![](http://www.machinelearning.ru/mimetex/?w_j) — вес j-го признака, ![](http://www.machinelearning.ru/mimetex/?w_0) — порог принятия решения, w — вектор весов. Предполагается, что искусственно введён «константный» нулевой признак: ![](http://www.machinelearning.ru/mimetex/?f_{0}(x)=-1). 

Величина ![](http://latex.codecogs.com/svg.latex?M_i%28w%29%3Dy_i%5Clangle%20x_i%2Cw%20%5Crangle)называется __отступом__ объекта относительно алгоритма классификации. Если ![](http://latex.codecogs.com/svg.latex?M_i%28w%29%3C0),алгоритм совершает на объекте ![](http://latex.codecogs.com/svg.latex?x_i) ошибку.

![](http://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D%28M%29)– монотонно невозрастающая __функция потерь__, мажорирует пороговую функцию
![](http://latex.codecogs.com/svg.latex?%5BM%3C0%5D%20%5Cleq%20%5Cmathcal%7BL%7D%28M%29).

Тогда __минимизацю суммарных потерь__ можно рассматривать как функцию вида
![](https://latex.codecogs.com/svg.latex?%5Ctilde%7BQ%7D%28w%2CX%5E%5Cell%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7B%5Cell%7DL%5Cmathcal%28M_i%28w%29%29%5Crightarrow%20%5Cmin_w)

### Метод стохастического градиента
Пусть задана обучающая выборка ![](https://latex.codecogs.com/gif.latex?X%5El%20%3D%20%5C%7B%28x_i%2C%20y_i%29%5C%7D_%7Bi%3D1%7D%5El%20%2C%20x_i%20%5Cin%20%5Cmathbb%7BR%7D%5En%20%2C%20y_i%20%5Cin%20%5C%7B-1%2C%20&plus;1%20%5C%7D)

Требуется найти вектор параметров ![](https://latex.codecogs.com/gif.latex?w%20%5Cin%20%5Cmathbb%7BR%7D) , при котором достигается минимум аппроксимированного эмпирического риска:

![](https://latex.codecogs.com/gif.latex?Q%28w%2C%20X%5El%29%20%3D%20%5Csum_%7Bi%3D1%7D%5El%20L%28%5Clangle%20w%2C%20x_i%20%5Crangle%20y_i%29%20%5Crightarrow%20%5Cmin_w)

Применим для минимизации Q(w) метод градиентного спуска. В этом методе выбирается некоторое начальное приближение для w, затем запускается итерационный процесс, на каждом шаге которого вектор w изменяется в направлении наиболее быстрого убывания функционала Q. Это направление противоположно направлению вектора градиента ![](https://latex.codecogs.com/gif.latex?Q%27%28w%29%20%3D%20%5Cleft%28%5Cdfrac%7B%5Cpartial%20Q%28w%29%7D%7B%5Cpartial%20w_j%7D%5Cright%29_%7Bj%3D1%7D%5En) :

![](https://latex.codecogs.com/gif.latex?w%20%3D%20w%20-%20%5Ceta%20Q%27%28w%29)
где ![](https://latex.codecogs.com/gif.latex?%5Ceta%20%3E%200) - темп обучения

Предположим, что функция L дифференцируема. Выпишем градиент:

![](https://latex.codecogs.com/gif.latex?w%20%3D%20w%20-%20%5Ceta%20%5Csum_%7Bi%3D1%7D%5En%20L%27%28%5Clangle%20w%2C%20x_i%20%5Crangle%20y_i%29%20x_i%20y_i)

Каждый прецедент ![](https://latex.codecogs.com/gif.latex?%28x_i%2C%20y_i%29) вносит аддитивный вклад в изменение вектора w, но вектор w изменяется только после перебора всех l объектов. Сходимость итерационного процесса можно улучшить, если выбирать прецеденты ![](https://latex.codecogs.com/gif.latex?%28x_i%2C%20y_i%29) по одному, для каждого делать градиентный шаг и сразу обновлять вектор весов:
![](https://latex.codecogs.com/gif.latex?w%20%3D%20w%20-%20%5Ceta%20L%27_a%20%28%5Clangle%20w%2C%20x_i%20%5Crangle%20y_i%29x_i%20y_i)

В методе __стохастического градиента (stochastic gradient, SG)__ прецеденты перебираются в случайном порядке. Если же объекты предъявлять в некотором фиксированном порядке, процесс может зациклиться или разойтись.Инициализация весов может производиться различными способами. Стандартная рекомендация - взять небольшие случайные значения ![](https://latex.codecogs.com/gif.latex?w_j%20%3D%20random%5Cleft%28%5Cfrac%7B-1%7D%7B2n%7D%3B%20%5Cfrac%7B1%7D%7B2n%7D%5Cright%29)

### Преимущества
- Метод легко реализуется и легко обобщается на нелинейные классификаторы и на нейронные сети ? суперпозиции линейных классификаторов.
- Метод подходит для динамического обучения, когда обучающие объекты поступают потоком, и вектор весов обновляется при появленикаждого объекта.
- Метод позволяет настраивать веса на избыточно больших выборках, за счет того, что случайной подвыборки может оказаться достаточно для обучения.

### Недостатки
- Функционал Q, как правило, многоэкстремальный, и процесс может сходиться к локальному минимуму, сходиться очень медленно или не сходиться вовсе.
- При большой размерности пространства n или малой длине выборки l возможно переобучение. При этом резко возрастает норма вектора весов, появляются большие по абсолютной величине положительные и отрицательные веса, классификация становится неустойчивой -- малые изменения обучающей выборки, начального приближения, порядка предъявления объектов или параметров алгоритма могут сильно изменить результирующий вектор весов, увеличивается вероятность ошибочной классификации новых объектов.
- Если функция потерь имеет горизонтальные асимптоты, то процесс может попасть в состояние ?параличаа. Чем больше значение скалярного произведения  , тем ближе значение производной L' к нулю, тем меньше приращение весов. Если веса w попали в область больших значений, то у них практически не остается шансов выбраться из этой мертвой зоныы.

Реализация:
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

## Адаптивный линейный элемент
Имеет _квадратичную функцию потерь_
![](http://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D%28M%29%3D%28M-1%29%5E2%3D%28%5Clangle%20w%2Cx_i%20%5Crangle%20y_i-1%29%5E2)

и _дельта-правило_ правило обновления весов
![](http://latex.codecogs.com/svg.latex?w%3Dw-%5Ceta%28%5Clangle%20w%2Cx_i%20%5Crangle-y_i%29x_i).

Реализация обучения:
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
Пример на ирисах Фишера с 2 классами (удален virginica). 
![lic](https://raw.githubusercontent.com/bestful/ML/master/samples/adaline.png)
Синей линией обозначена разделяющая поверхность на последней итерации

## Персептрон Розенблатта
Имеет _кусочно-линейную функцию потерь_
![](http://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D%3D%28-M%29_&plus;%3D%5Cmax%28-M%2C0%29)

и _правило Хебба_ для обновления весов
![](http://latex.codecogs.com/svg.latex?%5Ctext%7Bif%20%7D%5Clangle%20w%2Cx_i%20%5Crangle%20y_i%3C0%20%5Ctext%7B%20then%20%7D%20w%3A%3Dw&plus;%5Ceta%20x_iy_i).

Реализация обучения:
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
### Логистическая регрессия

Также является __оптимальный байесовским классификатором__ из-за своих довольно
сильных вероятностных предположений.

Имеет _логистическую функцию потерь_
![](http://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D%28M%29%20%3D%20%5Clog_2%281%20&plus;%20e%5E%7B-M%7D%29)

и _логистическое_ правило обновления весов
![](http://latex.codecogs.com/svg.latex?w%20%3A%3D%20w&plus;%5Ceta%20y_ix_i%5Csigma%28-%5Clangle%20w%2Cx_i%20%5Crangle%20y_i%29) , где
![](http://latex.codecogs.com/svg.latex?%5Csigma%28z%29%3D%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-z%7D%7D) – _сигмоидная функция_.
Реализация обучения:
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
### Адалайн, перцептрон Розенблатта и логистическая регрессия
![lic](https://raw.githubusercontent.com/bestful/ML/master/samples/lin_all.png)
## SVM 
Пусть выборка линейно разделима и существуют значения параметров ![](https://latex.codecogs.com/gif.latex?w_0) и ![](https://latex.codecogs.com/gif.latex?w_0) при которых функционал числа ошибок

![](https://latex.codecogs.com/gif.latex?Q%28w%2C%20w_0%29%20%3D%20%5Csum_%7Bi%3D1%7D%5El%20%5By_i%28%5Clangle%20w%2C%20x_i%5Crangle%20-%20w_0%29%20%5Cleq%200%5D)

принимает нулевое значение. Но тогда разделяющая гиперплоскость не единственна. Будем выбирать ее таким образом, чтобы она тстояла __максимально далеко от ближайших к ней точек обоих классов__. 

Заметим, что алгоритм a(x) не изменится, если ![](https://latex.codecogs.com/gif.latex?w_0) и ![](https://latex.codecogs.com/gif.latex?w_0) одновременно умножить на одну и ту же положительную константу. Выберем эту константу так, чтобы выполнялось условие

![](https://latex.codecogs.com/gif.latex?%5Cmin_%7Bi%3D1%2C%20%5Cdots%2C%20l%7D%20y_i%20%28%5Clangle%20w%2C%20x_i%20%5Crangle%20-%20w_0%29%20%3D%201)

Множество точек ![](https://latex.codecogs.com/gif.latex?%5C%7B%20x%3A%20-1%20%5Cleq%20%5Clangle%20w%2C%20x%20%5Crangle%20-w_0%20%5Cleq%201%20%5C%7D) описывает полосу, разделяющую классы.
![](https://raw.githubusercontent.com/bestful/ML/master/readme/SVM_margins.png)

Разделяющая гиперплоскость проходит ровно по середине между ними. Объекты, ближайшие к разделяющей гиперплоскости, лежат на границах полосы, и именно на них достигается минимум. В каждом из классов имеется хотя бы один такой объект.
Итак, в случае линейно разделимой выборки получаем задачу квадратичного программирования:

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bcases%7D%20%5Clangle%20w%2C%20w%20%5Crangle%20%5Crightarrow%20min%20%5C%5C%20y_i%28%5Clangle%20w%2C%20x_i%20%5Crangle%20-%20w_0%29%20%5Cgeq%201%2C%20i%20%3D%201%2C%20%5Cdots%2C%20l%20%5Cend%7Bcases%7D)

Обобщим постановку задачи на случай линейно неразделимой выборки: разрешим алгоритму допускать ошибки на обучающих объектах, но так, чтобы ошибок было как можно меньше. 

Для этого введем дополнительные переменные ![](https://latex.codecogs.com/gif.latex?%5Cepsilon_i%20%5Cgeq%200), характеризующие величину ошибки на объектах обучения ![](https://latex.codecogs.com/gif.latex?x_i) Тогда вместо предыдущей задачи будем иметь задачу

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bcases%7D%201/2%20%5Clangle%20w%2C%20w%20%5Crangle%20&plus;%20C%20%5Csum_%7Bi%3D1%7D%5El%20%5Cepsilon_i%20%5Crightarrow%20%5Cmin_%7Bw%2C%20w_0%2C%20%5Cepsilon%7D%20%5C%5C%20y_i%20%28%5Clangle%20w%2C%20x_i%5Crangle%20-%20w_0%29%20%5Cgeq%201%20-%20%5Cepsilon_i%20%5Cend%7Bcases%7D)

Положительная константа C является управляющим параметром метода и позволяет находить компромисс между максимизацией ширины разделяющей полосы и минимизацией суммарной ошибки.

___Пример SVM___
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
