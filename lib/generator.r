ML_GENERATOR <- T

gen <- function(f, evalf){
  genf <- f
  en <- environment(genf)
  fo <- formals(genf)
  evalf()
  genf
}