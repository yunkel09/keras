#   ____________________________________________________________________________
#   Redes Neuronales                                                        ####

# Uno podría pensar entonces que cuanto más neuronas utilicemos, mejor será el
# modelo, podríamos detectar una estructura más fina en sus datos. Sin embargo,
# cuantas más neuronas ocultas se utilicen, más probable es el riesgo de ajuste
# excesivo. El sobreajuste se produce porque con más neuronas hay una mayor
# probabilidad de que el red aprenda tanto los patrones como el ruido, en lugar
# de la estructura estadística subyacente de los datos. El resultado es una red
# con un buen desempeño en la muestra pero un mal desempeño fuera de la muestra.
 
# puede ser recomendable comenzar con redes de dos capas como máximo. 

# El truco fundamental en el aprendizaje de las redes neuronales es reducir los
# errores de predicción ajustando los valores de los pesos de la red. En el
# descenso de gradiente se intenta alcanzar el mínimo de la función de
# error/pérdida/costo/objetivo con respecto a los pesos de la red utilizando las
# derivadas calculadas en la retropropagación.
 
## Conjunto de entrenamiento y prueba

# El objetivo final de una red neuronal es entrenar a la red en datos de
# entrenamiento, con la expectativa de que, dados los nuevos datos de prueba la
# red neuronal podrá predecir sus salidas con precisión. Esta capacidad para
# predecir nuevas observaciones se llama generalización. Por lo tanto, para
# desarrollar una red neuronal necesitamos crear un conjunto de datos de
# entrenamiento y un conjunto de datos de prueba.

# Para evaluar la capacidad de generalización del modelo, vamos a comparar el
# rendimiento (error) del modelo sobre ambos conjuntos. Esperamos ver una ligera
# disminución del rendimiento del modelo desde el entrenamiento hasta la prueba.

## Sobreajuste

# Ya hemos mencionado el ajuste insuficiente (subajuste) y excesivo
# (sobreajuste) al elegir modelos. Una red neuronal que presenta un sobreajuste
# es una red neuronal en la que la tasa de error del conjunto de datos de
# entrenamiento es muy alta, y muy distinta de la tasa de error en el conjunto
# de datos de prueba. En este caso el modelo se ajusta demasiado a los datos de
# entrenamiento (es decir, modela el ruido que solo aparece allí en lugar de una
# señal verdadera).

## Hiperparámetros de redes neuronales

# Los hiperparámetros inciden sobre qué tan bien las redes neuronales son
# capaces de aprender, son configuraciones que se modifican para evaluar cómo
# funciona una red neuronal. Una mala selección de hiperparámetros puede
# conducir a redes neuronales con gran error, que no converjan, o convergen
# demasiado rápido en óptimos locales y no globales. Hemos visto varios ejemplos
# de hiperparámetros anteriormente, como la tasa de aprendizaje en
# retropropagación y la selección del SSE como métrica de rendimiento.

# Al evaluar los hiperparámetros de una red neuronal, generalmente comparamos
# varias redes neuronales creadas con diferentes hiperparámetros entrenados en
# el conjunto de datos de entrenamiento. Luego, cada una de ellas se evalúa en
# un conjunto de datos de prueba. La red con el error de conjunto de datos de
# prueba más bajo es la red neuronal con la mejor capacidad para generalizar a
# nuevas observaciones. Dadas dos redes neuronales con igual rendimiento (error)
# en el conjunto de datos de prueba, elegiríamos el modelo más simple, si no
# tenemos información adicional.

# Los valores óptimos de los hiperparámetros dependen de los conjuntos de datos
# específicos que se analizan, por lo tanto, en la mayoría de las redes
# neuronales los hiperparámetros deben "ajustarse" para obtener el mejor
# rendimiento.

## Ancho de la red neuronal

# Se puede diseñar una red neuronal con mejor rendimiento en el conjunto de
# datos de entrenamiento aumentando el ancho o la profundidad de la red neuronal
# (es decir, haciendo que la red más compleja). Este hiperparámetro permite un
# ajuste de la capacidad de la red neuronal. Equivale a ajustar polinomios de
# orden superior en regresión lineal para que coincidan perfectamente con la
# salida en función de los predictores. Sin embargo, con varias capas de
# neuronas, es probable que la red neuronal muestre un ajuste sobreajuste. Se
# prefieren los modelos sencillos hasta que la estructura de datos justifique un
# modelo más complejo.

## Funciones de activación

# Una función de activación es una función matemática que convierte la entrada
# en una salida y agrega la magia del procesamiento de la red neuronal. Sin
# funciones de activación, el funcionamiento de las redes neuronales será como
# funciones lineales, un polinomio de grado uno (una recta), donde la salida es
# directamente proporcional a la entrada.

# Sin embargo, la mayoría de los problemas que las redes neuronales intentan
# resolver son no lineales y de naturaleza compleja. Para lograr la no
# linealidad, se utilizan las funciones de activación.

# Algunas funciones de activación que podemos utilizar es:
	
#	La función sigmoidea es la función de activación más utilizada, pero adolece
#	de los siguientes contratiempos: Dado que utiliza un modelo logístico, los
#	cálculos son largos y complejos. Hace que los gradientes desaparezcan y no
#	pasen señales a través de las neuronas en algún momento. Además, es lenta en
#	convergencia, no está centrado en cero. Es popular en parte porque se puede
#	diferenciar fácilmente y por lo tanto reduce el costo computacional durante el
#	entrenamiento.
	
# La función tangente hiperbólica, también conocida como tanh, es muy similar a
# la función sigmoidea; sin embargo, el límite inferior de la curva está en un
# espacio negativo para manejar mejor los datos que contienen valores negativos.
# Dado que tanh está acotado entre -1 y 1, el gradiente es más grande y la
# derivada es más pronunciada. Estar delimitado significa que tanh se centra
# alrededor de 0, lo que puede ser ventajoso en un modelo con una gran cantidad
# de capas ocultas, ya que los resultados de una capa son más fáciles de usar
# para la siguiente capa.

# La función de unidades lineales rectificadas (ReLU) es una función híbrida que
# se ajusta a una línea para valores positivos de x mientras asigna cualquier
# valor negativo de x con un valor de 0. Aunque la mitad de esta función es
# lineal, la forma es no lineal y lleva consigo todas las ventajas de la no
# linealidad, como la posibilidad de utilizar la derivada para retropropagación.
# A diferencia de las dos funciones de activación anteriores, no tiene límite
# superior. Esta falta de una restricción puede ser útil para evitar el problema
# con la función sigmoidea o tanh, donde el gradiente se vuelve muy gradual
# cerca de los extremos y proporciona poca información para ayudar al modelo a
# seguir aprendiendo.

# Otra ventaja importante de ReLU es cómo conduce al cero en la red neuronal
# debido a la caída en el punto central. Usando signoid o tanh, muy pocos
# valores de salida de la función serán cero, lo que significa que las funciones
# de activación se activarán, lo que conducirá a una red densa. Por el
# contrario, ReLU da como resultado muchos más valores de salida de cero, lo que
# lleva a que se activen menos neuronas y a una red mucho más dispersa. ReLU
# aprenderá más rápido que sigmoide y tanh debido a su simplicidad.

# Elegir la función de activación correcta. En la mayoría de los casos, siempre
# debemos considerar a ReLU primero. Pero tenga en cuenta que ReLU solo debe
# aplicarse a capas ocultas.


##  ............................................................................
##  Paquetes                                                                ####

import::from(magrittr, "%T>%", "%$%",  "%<>%", .into = "operadores")
import::from(parallel, detectCores, makePSOCKcluster, stopCluster)
import::from(doParallel, registerDoParallel)
import::from(conectigo, cargar_fuentes)
import::from(GGally, ggpairs, wrap)
import::from(janitor, make_clean_names)
pacman::p_load(finetune, parsnipExtra, tensorflow, keras, tictoc, tidymodels,
					tidyverse)

options(pillar.sigfig    = 5,
		  tibble.print_min = 10,
		  scipen = 999,
		  digits = 7,
		  readr.show_col_types = FALSE,
		  dplyr.summarise.inform = FALSE)

##  ............................................................................
##  Funciones                                                               ####

loess_lm <- function(data, mapping, ...){
	
	ggplot(data = data, mapping = mapping) + 
		geom_point(alpha = 0.9) + 
		stat_smooth(formula = y ~ x, 
						method = "lm", 
						se = TRUE, 
						color = "red",
						fill = "red",
						size = 0.5, 
						alpha = 0.2,
						linetype = "longdash", 
						...)
}


unregister <- function() {
	env <- foreach:::.foreachGlobals
	rm(list=ls(name=env), pos=env)
}

# fuentes del paquete conectigo
cargar_fuentes()


# tema con grid horizontal y vertical
drako <- theme_bw(base_family = "yano", base_size = 14) +
	theme(plot.margin    = unit(c(6, 1, 1, 1), "mm"),
			axis.title     = element_text(size = 12),
			axis.text      = element_text(size = 12),
			plot.title     = element_text(size = 18),
			plot.subtitle  = element_text(size = 12))


##  ............................................................................
##  Ejercicios obligatorios                                                 ####

# JERCICIO 1. datos College
# Analiza la clasificación entre colegios privados y públicos de las
# universidades de EE. UU. según varios estadísticos de desempeño. Los datos se
# encuentran en el objeto College del paquete ISLR y corresponden a la edición
# de 1995 de US News y World Report.
 
college <- ISLR::College |> 
 as_tibble(.name_repair = make_clean_names)


# EJERCICIO 2. datos Auto
# Analiza el consumo de gasolina para 392 vehículos según sus características.
# Los datos se encuentran en el objeto Auto del paquete ISLR y fueron utilizados
# en 1983 en la American Statistical Association Exposition.
 
autos <- ISLR::Auto |> 
	as_tibble()

# En ambos ejercicios sigue los siguientes pasos:
# Observa y grafica los datos. 
# Transforma los datos cuando sea necesario. 
# Divide el conjunto de datos en uno de entrenamiento y otro de prueba. 
# Construye el modelo NN, grafica e interpreta el resultado. 
# Evalúa la performance del modelo NN. 

##  ............................................................................
##  Boston                                                                  ####

# regresión
boston <- MASS::Boston |> 
 as_tibble() |>
 select(crim, indus, nox, rm, age, dis, tax, ptratio, lstat, medv)

summary(boston)


### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Prep                                                                    ####

boston_std <- recipe(medv ~ ., data = boston) |> 
	step_range(all_predictors()) |>   # método mínimo-máximo
	prep() |>
	juice()

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Explorar                                                                ####

# boston_std |> 
#  ggpairs(data = _, lower = list(continuous = loess_lm),
#          upper = list(continuous = wrap("cor", size = 5))) 

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Interacciones                                                           ####

# Más formalmente, se dice que dos o más predictores interactúan si su efecto
# combinado es diferente (menor o mayor) de lo que esperaríamos si tuviéramos
# que agregar el impacto de cada uno de sus efectos cuando se consideran solos.

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Muestreo                                                                ####

boston_split <- initial_split(boston, prop = 0.7)
boston_train <- training(boston_split)
boston_test  <-  testing(boston_split)

boston_train

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Preprocesamiento                                                        ####

# aplicar `step_YeoJohnson()` antes de normalizar

# No siempre es necesario transformar los datos numéricos, sin embargo, se ha
# demostrado que cuando los valores numéricos se normalizan, la formación de
# redes neuronales suele ser más eficiente y conduce a una mejor predicción.

# Muchos de los predictores tienen distribuciones sesgadas. Dado que PCA se basa
# en la varianza, los valores extremos pueden tener un efecto perjudicial en
# estos cálculos. Para contrarrestar esto, agreguemos un paso de receta que
# estime una transformación de Yeo-Johnson para cada predictor (Yeo y Johnson
# 2000). Si bien originalmente se pensó como una transformación del resultado,
# también se puede usar para estimar transformaciones que fomenten
# distribuciones más simétricas. Este paso step_YeoJohnson() ocurre en la receta
# justo antes de la normalización inicial a través de step_normalize(). Luego,
# combinemos esta receta de ingeniería de características con nuestra
# especificación de modelo de red neuronal mlp_spec.

# receta básica solo predictores normalizados
solo_normalizado <- boston_train |> 
 recipe(medv ~ .) |> 
 step_normalize(all_predictors())

# primero un corrección del sesgo y luego normalizar
normalizado_yeoj <- boston_train |> 
 recipe(medv ~ .) |> 
 step_YeoJohnson(all_numeric_predictors()) |> 
 step_normalize(all_predictors())

solo_normalizado|> tidy()

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Modelos                                                                 ####

show_engines("mlp")

# nnet: tenga en cuenta que parsnip establece automáticamente la activación
# lineal en la última capa.
mlp_nnets <- mlp(hidden_units = tune(),
					  penalty      = tune(),
					  epochs       = tune()) |>
	set_engine("nnet") |> 
	set_mode("regression")


# keras
mlp_keras <- mlp(hidden_units = tune(), 
					  penalty      = 0,
					  epochs       = 500,
					  activation   = "softmax") |> 
	set_engine('keras') %>%
	set_mode('regression')


# hidden_units: el número de unidades ocultas,
# penalty:      la cantidad de penalización por caída de peso.
# epochs:       el número de épocas/iteraciones de ajuste en el entrenamiento
#               del modelo.
# trace = 0:    evita el registro adicional del proceso de entrenamiento.
# MaxNWts:      el número máximo permitido de pesos. No hay un límite intrínseco
#               en el código, pero aumentar MaxNWts probablemente permitirá
#               ajustes que son # muy lentos y consumen mucho tiempo.


# evaluar los posibles valores de los hiperparámetros de los modelos:
mlp_param <- mlp_nnets |> extract_parameter_set_dials()
mlp_param |> extract_parameter_dials("hidden_units")
mlp_param |> extract_parameter_dials("penalty")
mlp_param |> extract_parameter_dials("epochs")
# mlp_param |> extract_parameter_dials("activation")

# Esta salida indica que los objetos de parámetro están completos e imprime sus
# rangos predeterminados. Estos valores se utilizarán para demostrar cómo crear
# diferentes tipos de cuadrículas de parámetros.

# lasso
lasso_reg <- linear_reg(penalty = tune(), mixture = 1) |> 
 set_mode("regression") |> 
 set_engine("glmnet")

# penalty = cuantas variables se van a remover
# mixture = decidir si queremos hacer una regularización de tipo L1 o L2. en
#           este caso como lo que queremos es un lasso, seleccionamos 1. si
#           quisiéramos un `ridge regression` seleccionaríamos un 0, y si
#           quisiéramos un `elastic net` le ponemos `tune()`

# ver los parámetros principales del modelo
lasso_reg |> extract_parameter_set_dials()

# revisar específicamente cuanto es el rango de penalización por defecto
lasso_param |> extract_parameter_dials("penalty")

# actualicemos para que la penalización vaya de 0 a 1
lasso_param <- lasso_reg |> extract_parameter_set_dials() |> 
 update(penalty = penalty(range = c(0, 1)))

# validar que la actualización del parámetro penalty se realizó correctamente
lasso_param |> extract_parameter_dials("penalty")

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Cross-validación                                                        ####

boston_folds <- vfold_cv(boston_train, strata = medv)

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Listas                                                                  ####

# establecer lista de recetas de preprocesamiento y modelos candidatos

recetas <- list(normalizado = solo_normalizado, 
					 sesgo_yeojh = normalizado_yeoj)

# motores de redes neuronales candidatos
modelos_sin_gpu <- list(nnets = mlp_nnets, lasso = lasso_reg)
modelos_con_gpu <- list(keras = mlp_keras)

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Workflows                                                               ####


# workflow con motores que no requieren uso de GPU
wflow_sin_gpu <- workflow_set(preproc = recetas, 
										models  = modelos_sin_gpu,
										cross   = TRUE)


# actualicemos los parámetros de penalización para lasso
wflow_sin_gpu %<>%
 option_add(param_info = lasso_param, id = "normalizado_lasso") %<>%
 option_add(param_info = lasso_param, id = "sesgo_yeojh_lasso")


# workflow con motores que si requiere uso de GPU
wflow_con_gpu <- workflow_set(preproc = recetas, 
										models  = list(keras = mlp_keras))


# ver contenido del los workflows
wflow_sin_gpu
wflow_con_gpu


# verificar uno de los modelos
wflow_sin_gpu |> extract_workflow(id = "normalizado_nnets")

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Tuning                                                                  ####

# no es necesario definirlas
metricas <- metric_set(rmse, rsq)

# con los métodos de carrera, el proceso de ajuste evalúa todos los modelos
# en un subconjunto inicial de remuestreos. En función de sus métricas de
# rendimiento actuales, algunos conjuntos de parámetros no se consideran en
# remuestreos posteriores.

# método race
race_ctrl <- control_race(
	save_pred     = TRUE,
	parallel_over = "everything",
	save_workflow = TRUE)


# habilitar procesamiento paralelo para usar todos los núcleos del CPUs
all_cores <- detectCores(logical = FALSE)
clusterpr <- makePSOCKcluster(all_cores)
registerDoParallel(clusterpr)


# nnets: 26.15 segundos
tic()
grid_sin_gpu <- wflow_sin_gpu |> 
	workflow_map(
		"tune_race_anova",
		seed = 1503,
		resamples = boston_folds,
		verbose = TRUE,
		grid    = 20,
		control = race_ctrl, 
		metrics = metricas)
toc()

# detener paralización
stopCluster(clusterpr)
unregister()


# keras
tic()
# 2 recetas -> 310 seg = 5 min
grid_gpu <- wflow_con_gpu |> 
	workflow_map(
		"tune_race_anova",
		seed = 1503,
		resamples = boston_folds,
		verbose = TRUE,
		grid    = 25,
		control = race_ctrl, 
		metrics = metricas)
toc()


### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Revisión                                                                ####

nrow(collect_metrics(grid_sin_gpu, summarize = FALSE)) # 872
nrow(collect_metrics(grid_gpu, summarize = FALSE))

grid_sin_gpu
grid_gpu

# unificar
tune_res <- bind_rows(grid_sin_gpu, grid_gpu)

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Extracciones                                                            ####

# extracciones
grid_sin_gpu |> extract_workflow_set_result(id = "normalizado_lasso")


# seleccionar el mejor con base a una métrica
grid_sin_gpu |> extract_workflow_set_result(id = "normalizado_nnets") |> 
 select_best(metric = "rmse")

# revisar las notas
grid_sin_gpu |>
 extract_workflow_set_result(id = "normalizado_nnets") |> 
 collect_notes()

# Esto puede ocurrir cuando el modelo predice un valor único para todas las
# muestras. Dos ejemplos podrían ser un modelo regularizado que elimine todos
# los predictores excepto el intercepto y un árbol CART que no contenga
# divisiones.


grid_sin_gpu |> extract_preprocessor(id = "normalizado_nnets")
grid_sin_gpu |> extract_spec_parsnip(id = "normalizado_nnets")
grid_sin_gpu |> extract_workflow(id = "normalizado_nnets")

# revisar métricas
grid_sin_gpu |>
 extract_workflow_set_result(id = "normalizado_nnets") |> 
 collect_metrics()

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Evaluación                                                              ####

grid_sin_gpu |> 
 rank_results() |> 
 filter(.metric == "rmse") |> 
 select(model, .config, rmse = mean, rank)

tune_res |> 
 rank_results() |> 
 filter(.metric == "rmse") |> 
 select(model, .config, rmse = mean, rank)

top_models <- grid_sin_gpu |> 
 rank_results(select_best = TRUE, rank_metric = "rmse") |> 
 select(modelo = wflow_id, .metric, mean, rank) |> 
 pivot_wider(names_from = .metric, values_from = mean)

top_models

# RMSE: La idea básica es medir qué tan malas/erróneas son las predicciones del
# modelo en comparación con los valores reales observados. Entonces, un RMSE
# alto es "malo" y un RMSE bajo es "bueno".

# el error cuadrático medio (RMSE) es una métrica de rendimiento común que se
# utiliza en modelos de regresión. Utiliza la diferencia entre los valores
# observados y predichos en sus cálculos

# El error cuadrático medio (RMSE) es la desviación estándar de los residuos
# (errores de predicción). Los residuos son una medida de qué tan lejos están
# los puntos de datos de la línea de regresión; RMSE es una medida de cuán
# dispersos están estos residuos. En otras palabras, le dice qué tan
# concentrados están los datos alrededor de la línea de mejor ajuste

autoplot(
	grid_sin_gpu,
	rank_metric = "rmse",  
	metric = "rmse",       
	select_best = TRUE) +
	geom_text(aes(y = mean - 1.2, label = wflow_id, color = wflow_id), 
				 angle = 90, hjust = 1) +
	lims(y = c(-5, 10)) +
	drako

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Hiperparámetros                                                         ####

autoplot(grid_sin_gpu, id = "normalizado_nnets", metric = "rmse")


### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Finalizar                                                               ####

nombre_mejor_modelo <- top_models |> 
 filter(rank == 1) |> 
 pull(modelo)

# seleccionar el mejor modelo
mejor_modelo <- grid_sin_gpu %>% 
	extract_workflow_set_result(nombre_mejor_modelo) %>% 
	select_best(metric = "rmse")

mejor_modelo


boosting_test_results <- grid_sin_gpu %>% 
	extract_workflow("normalizado_nnets") %>% 
	finalize_workflow(best_results) %>% 
	last_fit(split = concrete_split)


