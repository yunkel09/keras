
import::from(magrittr, "%T>%", "%$%", .into = "operadores")
import::from(lubridate, .except = c("intersect", "setdiff", "union"))
pacman::p_load(inspectdf, embed, textrecipes, janitor, tidymodels, tidyverse)

options(pillar.sigfig    = 5,
		  tibble.print_min = 10,
		  scipen = 999,
		  digits = 7,
		  readr.show_col_types = FALSE,
		  dplyr.summarise.inform = FALSE)

# clasificación: private
coll <- ISLR::College |> as_tibble(.name_repair = make_clean_names)

# regresión: mpg
auto_raw <- ISLR::Auto |> as_tibble()

ver <- . %>% prep() %>% juice()

# separar el feature *name* en marca, modelo y trim
# La moldura es cómo se viste el automóvil, se hace para que se vea bien y
# también afecta la comodidad del vehículo. Al fabricante le gusta agrupar estas
# características para que sean más fáciles de construir. Entonces, si desea
# asientos de cuero, es posible que deba comprar un paquete de lujo que incluya
# otras características, como iluminación adicional y opciones de radio. Si
# quieres gastar menos, un modelo más básico no tendrá esas características,
# pero costará menos dinero.
# 
# https://bit.ly/3GnHJ9T
# La versión básica, o el modelo básico, no suele tener mucho equipamiento, pero
# es la más asequible. Cada nivel de equipamiento después de este ofrece más
# características y opciones hasta que llegue al nivel de equipamiento más alto.

# supuesto: podemos asumir que si un vehículo en particular no cuenta con una
# descripción de su trim (acabado) entonces se refiere al modelo "base" 
# https://bit.ly/3GnHJ9T 

# El nivel de equipamiento del modelo base de un fabricante de automóviles es la
# versión más simple del nuevo vehículo. Un modelo base representa la variación
# de modelo menos costosa del vehículo que ofrece el fabricante de automóviles.
# **Es posible que el nivel de equipamiento de un modelo base no tenga un nombre
# específico**, pero el mismo modelo de automóvil puede tener varios niveles de
# equipamiento. Un gran ejemplo de esto es el Nissan Altima, que es un modelo
# base sin nombre pero tiene niveles de equipamiento *adicionales*: S, SR, SV y
# SL. Por otro lado, el Toyota Camry LE es un modelo base con un nivel de
# equipamiento con nombre. El SE, XSE y XLE son paquetes adicionales de
# equipamiento del Toyota Camry. (https://bit.ly/3wyko1N)

autos <- auto_raw |> 
	mutate(across(name, as.character)) |> 
	separate(name, c("make", "model", "trim"),
				extra = "drop", fill = "right") |> 
	mutate(
		across(trim, replace_na, "base"),
		across(where(is.character), as.factor))
	
	
# gráfica exploratia
autos |> select(make:trim) |>
	inspect_cat() |> 
	show_plot(high_cardinality = 4)
	
# hay 172 posibles combinaciones. esto denota una **alta cardinalidad**
autos |>
	distinct(make, model, trim) %>%
	janitor::get_dupes(make, model, trim)


# data spending
autos_split <- initial_split(autos, prop = 0.3)
autos_train <- training(autos_split)


# hay muchas opciones para codificar variables categóricas de alta cardinalidad
# que el uso de variables dummy o indicadoras. un método es el llamado
# codificación de efectos el cual reemplaza la variable categórica con una
# única variable numérica que mide el efecto de esos datos. Por ejemplo, para
# el caso de la marca (make) podríamos computar la media o media de mpg para
# cada marca y sustituir esta media por los valores en los datos originales:
# 
# también le dicen “target encoding”.

autos |> 
	group_by(make) |> 
	summarise(media = mean(mpg), std_err = sd(mpg) / sqrt(length(mpg))) |> 
	ggplot(aes(y = reorder(make, media), x = media)) +
	geom_point() +
	geom_errorbar(aes(xmin = media - 1.64 * std_err, xmax = media + 1.64 * std_err)) +
	labs(y = NULL, x = "Millas por galón")


# ahora vemos como queda la columna *make* a utilizar una codificación de efectos
# con `step_lencode_glm()`. 
recipe(mpg ~ ., data = autos_train) |> 
	step_impute_mode(model) |> 
	step_other(model, trim, threshold = tune()) |> 
	# Estos pasos usan un modelo lineal generalizado para estimar el efecto de cada
	# nivel en un predictor categórico sobre el resultado.
	step_lencode_mixed(model, trim, outcome = vars(mpg)) |>
	step_unknown(all_nominal_predictors()) |> 
	step_other(make, threshold = 0.001) |> ver() |> count(make, sort = T) |> print(n = Inf)
	# step_other(make, threshold = tune()) |> ver() |> count(make, sort = T) |> print(n = Inf)
	step_dummy(make) |> ver()


# feature hashing
		
# Las variables ficticias tradicionales, como se describe en la Sección 8.4.1,
# requieren que se conozcan todas las categorías posibles para crear un
# conjunto completo de características numéricas. Los métodos hash de
# características (Weinberger et al. 2009) también crean variables ficticias,
# **pero solo consideran el valor de la categoría para asignarlo a un grupo
# predefinido de variables ficticias.** Veamos los valores de Neighborhood en
# Ames nuevamente y usemos la función rlang::hash() para comprender más:

# pareciera lo mismo que hacer variables dummies solo que en diferente orden
# https://youtu.be/XelrzDtEnPY
	
# lo que pasa es que se almacen la posición en la matriz
# 
# efecto avalancha (avalanch effect): es que no importa que tan parecida sean
# las cadenas, el hash resultante es bastante diferente.
	
	
# colisión: se han publicado muchos estudios sobre que tanto impacto tienen
# las colisiones. Podemos tener una gran cantidad de colisiones (categorías a
# las que se le asignó el mismo hash) digamos hasta el 50% y el desempeño no
# baja mucho.  La falta de interpretabilidad se debe a que como hay colisiones
# no es posible saber a que categoría realmente pertenece el peso que le
# estas dando (digamos en un modelo lineal)
# 
# es función lineal
	
data(ames)
set.seed(501)
ames_split <- initial_split(ames, prop = 0.80)
ames_train <- training(ames_split)

library(rlang)

ames_train |> count(Neighborhood)

ames_hashed <- ames_train |> 
	select(Neighborhood) |> 
	mutate(hash = map_chr(Neighborhood, hash))

	
ames_hashed

# Si ingresamos Briardale a esta función hash, siempre obtendremos el mismo
# resultado. Neighborhood en este caso se denominan "claves", mientras que
# las salidas son los "hashes".
	
ames_hashed |> filter(Neighborhood == "Gilbert")

# Una función hash toma una entrada de tamaño variable y la asigna a una salida
# de tamaño fijo. Las funciones hash se usan comúnmente en criptografía y bases
# de datos.

# En el hashing de características, el número de hashes posibles es un
# hiperparámetro y lo establece el desarrollador del modelo mediante el cálculo
# del módulo de los hashes enteros. Podemos obtener dieciséis valores hash
# posibles usando Hash %% 16:

cont <- ames_hashed |> 
	# primero haga un hash más pequeño para los enteros que R puede manejar
	# strtoi convierte cadenas a enteros dado un base utilizando la función en C
	# strtol
	mutate(hash = strtoi(substr(hash, 26, 32), base = 16L),
			 hash = hash %% 16)

# se asignó un hash a cada categoría
cont

# pero muchos categorías comparten varios hash, veamos el hash 7. esto se llama
# colisiones
cont |> 
	filter(hash == 7) |> 
	distinct(Neighborhood, .keep_all = T)
	

# vemos que hay 28 categorias distintas
ames_hashed |> count(Neighborhood)
ames_hashed |> distinct(Neighborhood)


# Hashing Trick

# El hashing de características se introdujo como un método de reducción de
# dimensionalidad con una premisa simple. Comenzamos con una función hash que
# luego aplicamos a nuestros tokens.

# Una función hash toma una entrada de tamaño variable y la asigna a una salida
# de un tamaño fijo. Las funciones hash se usan comúnmente en criptografía.

# Las funciones hash suelen ser muy rápidas y tienen ciertas propiedades. Por
# ejemplo, se espera que la salida de una función hash sea uniforme, con todo el
# espacio de salida lleno de manera uniforme. El "efecto de avalancha" describe
# cómo las cadenas similares se codifican de tal manera que sus hashes no son
# similares en el espacio de salida.

# Supongamos que tenemos muchos nombres de países en un vector de caracteres.
# Podemos aplicar la función hash a cada uno de los nombres de países para
# proyectarlos en un espacio entero definido por la función hash.

# Dado que hash() crea hashes que son muy largos, vamos a crear small_hash() con
# fines de demostración aquí que genera hashes ligeramente más pequeños. (Los
# detalles específicos de qué hashes se generan no son importantes aquí).


countries <- c("Palau", "Luxembourg", "Vietnam", "Guam", "Argentina",
					"Mayotte", "Bouvet Island", "South Korea", "San Marino",
					"American Samoa")

small_hash <- function(x) {
	strtoi(substr(hash(x), 26, 32), 16)
}

map_int(countries, small_hash)

# Nuestra función small_hash() usa 7 * 4 = 28 bits, por lo que el número de
# valores posibles es 2^28 = 268435456. Es cierto que esto no es una gran mejora
# con respecto a 10 nombres de países. Tomemos el módulo de estos valores
# enteros grandes para proyectarlos a un espacio más manejable.

map_int(countries, small_hash) %% 24

# estos valores resultantes se pueden utilizar como índices para crear una
# matriz


# Este método es muy rápido; tanto el hashing como el módulo se pueden realizar
# de forma independiente para cada entrada, ya que ninguno necesita información
# sobre el corpus completo. Dado que estamos reduciendo el espacio, existe la
# posibilidad de que varias palabras tengan el mismo valor. Esto se llama
# colisión y, a primera vista, parece que sería un gran problema para un modelo.
# Sin embargo, la investigación encuentra que el uso de hashing de
# características tiene aproximadamente la misma precisión que un modelo simple
# de bolsa de palabras, y el efecto de las colisiones es bastante menor (Forman
# y Kirshenbaum 2008).

# Otro paso que se toma para evitar los efectos negativos de las colisiones hash
# es usar una segunda función hash que devuelve 1 y −1. Esto determina si
# estamos sumando o restando el índice que obtenemos de la primera función
# hashin. Supongamos que las palabras "exterior" y "agradable" tienen un valor
# entero de 583. Sin el segundo hash, colisionarían en 2. Al usar hash firmado,
# tenemos un 50% de posibilidades de que se cancelen entre sí, lo que intenta
# detener una característica de crecer demasiado.

# las desventajas del hashing es qque:
# - todavía tiene un parámetro de tuneo
# - no puede ser revertido 



# opcion 1
recipe(mpg ~ ., data = autos_train) |> 
	step_impute_mode(model) |> 
	step_unknown(all_nominal_predictors()) |> 
 
	step_lencode_mixed(model, trim, outcome = vars(mpg)) |>
	# step_other(make, threshold = tune()) |>
   step_other(make, threshold = 0.001) |> 
	# ahora en lugar de tener 28 marcas tenemos 16 codificadas en hash
	# usaremos `signed = TRUE` para evitar colisiones
	step_dummy_hash(make, signed = F, num_terms = 16L) |>
	# algunas columnas creadas podrían contener solo ceros por lo que hay que
	# filtrar con step_zv
	step_zv(all_predictors()) |> 
	ver() |> 
	select(starts_with("dummy")) |> 
	suppressWarnings() |> 
	map_lgl(~ all(.x == 0))



# El hashing de características es rápido y eficiente, pero tiene algunas
# desventajas. Por ejemplo, **diferentes valores de categoría a menudo se asignan
# al mismo valor hash. Esto se llama colisión o aliasing.** ¿Con qué frecuencia
# sucedió esto con nuestros vecindarios en Ames? La Tabla 17.3 presenta la
# distribución del número de barrios por valor hash.

# Entity embeddings

# step_embed


recipe(mpg ~ ., data = autos_train) |> 
	step_mutate()









