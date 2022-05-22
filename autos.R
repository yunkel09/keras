
import::from(magrittr, "%T>%", "%$%", .into = "operadores")
import::from(lubridate, .except = c("intersect", "setdiff", "union"))
pacman::p_load(textrecipes, janitor, tidymodels, tidyverse)

options(pillar.sigfig    = 5,
		  tibble.print_min = 10,
		  scipen = 999,
		  digits = 7,
		  readr.show_col_types = FALSE,
		  dplyr.summarise.inform = FALSE)

# clasificación: private
coll <- ISLR::College |> as_tibble(.name_repair = make_clean_names)

# regresión: mpg
auto <- ISLR::Auto |> as_tibble()


au <- auto |> 
	mutate(across(name, as.character)) |> 
	separate(name, c("marca", "modelo", "linea"), extra = "drop", fill = "right")
	
library(scotus)

scotus_filtered |> 
	as_tibble()
	
set.seed(1234)
scotus_split <- scotus_filtered %>%
	mutate(year = as.numeric(year),
			 text = str_remove_all(text, "'")) %>%
	initial_split()

scotus_train <- training(scotus_split)
scotus_test <- testing(scotus_split)

scotus_train2 <- scotus_train |> sample_n(50)

ver <- . %>% prep() %>% juice()

# soctous_rec <- 
recipe(year ~ text, data = scotus_train2) |> 
	# 1: tokenizar (dividir en palabras todo el texto)
	step_tokenize(text) |>
	# 2: filtrar para mantener únicamente el top 1000 de tokens por frecuencia
	step_tokenfilter(text, max_tokens = 1e3) |> 
	# 3: ponderar cada frecuencia de token por la frecuencia inversa del
	# documento (tf-idf)
	step_tfidf(text) |>
	# 4: centrar y escalar los predictores
	step_normalize(all_predictors())