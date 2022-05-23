#   ____________________________________________________________________________
#   Text Analsys                                                            ####


# Para crear variables para el aprendizaje automático supervisado a partir del
# lenguaje natural, necesitamos alguna forma de representar el texto sin
# procesar como números para que podamos realizar cálculos sobre ellos.

library(tokenizers)
library(tidytext)
library(hcandersenr)

import::from(magrittr, "%T>%", "%$%", .into = "operadores")
import::from(lubridate, .except = c("intersect", "setdiff", "union"))
pacman::p_load(janitor, SnowballC, tidyverse)

options(pillar.sigfig    = 5,
		  tibble.print_min = 10,
		  scipen = 999,
		  digits = 7,
		  readr.show_col_types = FALSE,
		  dplyr.summarise.inform = FALSE)

#   ____________________________________________________________________________
#   Contenido                                                               ####


# tokenización,
# Remover stop words
# stemming
# lematización


 

##  ............................................................................
##  Tokenización                                                            ####

the_fir_tree <- hcandersen_en %>%
	filter(book == "The fir tree") %>%
	pull(text)

# dividir las palabras usando expresiones regulares no es muy conveniente
# se pierde información. por esta razón debemos utilizar la tokenización.
the_fir_tree[1:2] |> 
 strsplit("[^a-zA-Z0-9]+")


the_fir_tree[1:2] |> 
 tokenize_words()


# Pensar en un token como una palabra es una forma útil de comenzar a comprender
# la tokenización, incluso si es difícil de implementar de manera concreta en el
# software. Podemos generalizar la idea de un token más allá de una sola palabra
# a otras unidades de texto. Podemos tokenizar texto en una variedad de unidades
# que incluyen:

# caracteres, palabras, oraciones, lineas, párrafos y n-grama

# Un n-gramas es un conjunto de n elementos consecutivos en un documento de
# texto, que puede incluir palabras, números, símbolos y puntuación

# usar tidytext
sample_tibble <- tibble(texto = the_fir_tree[1:2])
 

sample_tibble |> 
 unnest_tokens(word, texto, token = "words", strip_punct = FALSE)

tft_token_characters <- tokenize_characters(x = the_fir_tree,
														  lowercase = TRUE,
														  strip_non_alphanum = TRUE,
														  simplify = FALSE)


# veamos cuales son las palabras más usada en cada cuento
hcandersen_en |> 
 filter(book %in% c("The fir tree", "The little mermaid")) |> 
 unnest_tokens(output = word, input = text) |> 
 count(book, word) |> 
 group_by(book) |> 
 arrange(desc(n)) |> 
 slice(1:5)

# las 5 palabras más comunes en cada cuento de hada son poco informativas, con
# la excepción de "tree".

# a estas palabras poco informativas son llamadas stop_words

## Tokenizar con n-grams

# Un n-grama (a veces escrito "ngrama") es un término en lingüística para una
# secuencia contigua de n elementos de una secuencia dada de texto o discurso.
# El elemento puede ser fonemas, sílabas, letras o palabras según la aplicación,
# pero cuando la mayoría de la gente habla de n-gramas, se refiere a un grupo de
# n palabras. En este libro, usaremos n-grama para denotar n-gramas de palabras
# a menos que se indique lo contrario.

# unigrama: "hello,", "day", "little"
# bigrama: "fir tree", "fresh air" "Robin Hood"
# trigram: "You and I"
 
# Los n-grams son frases. fácil!

 


tft_token_ngram <- tokenize_ngrams(x = the_fir_tree,
											  lowercase = TRUE,       # regresar en minúsculas
											  n = 3L,                   # el nivel de n-gram (trigrams)
											  n_min = 3L,              # número mínimo de n-gram
											  stopwords = character(),
											  ngram_delim = " ",
											  simplify = FALSE)


tft_token_ngram[[1]]

# Es importante elegir el valor correcto para n al usar n-gramas para la #
# pregunta que queremos responder

# nos interesa capturar el orden de las palabras

add_paragraphs <- function(data) {
	pull(data, text) %>%
		paste(collapse = "\n") %>%
		tokenize_paragraphs() %>%
		unlist() %>%
		tibble(text = .) %>%
		mutate(paragraph = row_number())
}

library(janeaustenr)

northangerabbey_paragraphed <- tibble(text = northangerabbey) |> 
 mutate(chapter = cumsum(str_detect(text, "^CHAPTER "))) |> 
 filter(chapter > 0, !str_detect(text, "^CHAPTER ")) |> 
 nest(data = text) |> 
 mutate(data = map(data, add_paragraphs)) |> 
 unnest(cols = c(data))


# ahora convertir the_fir_tree de un vector con una línea por elemento a un
# vector con una oración por elemento.

the_fir_tree_sentences <- the_fir_tree |> 
 paste(collapse = " ") |> 
 tokenize_sentences()

head(the_fir_tree_sentences[[1]])

# ahora convertiremos a una oración por elemento
hcandersen_senteces <- hcandersen_en |> 
 nest(data = text) |> 
 mutate(data = map_chr(data, ~ paste(.x$text, collapse = " "))) |> 
 unnest_sentences(sentences, data)

# ¿Dónde falla la tokenización?

l <- "Don’t forget you owe the bank $1 million for the house."

# vemos que elimina el signo de dolar
tokenize_words("$1")

tokenize_words("$1", strip_punct = FALSE)

# si removemos el punto al final no será posible encontrar la última palabra
# al utilizar n-gramas.


letter_tokens <- str_extract_all(
	string = "This sentence include 2 numbers and 1 period.",
	pattern = "[:alpha:]{1}"
)

letter_tokens

# palabras con guiones

str_split("This isn't a sentence with hyphenated-words.", "[:space:]")

str_split("This isn't a sentence with hyphenated-words.", "[:space:]") %>%
	map(~ str_remove_all(.x, "^[:punct:]+|[:punct:]+$"))

# solo nos devuelve las palabras que tiene un guion
str_extract_all(
	string = "This isn't a sentence with hyphenated-words.",
	pattern = "[:alpha:]+-[:alpha:]+"
)

# le agregamos el signo de ? para que incluya el resto.
str_extract_all(
	string = "This isn't a sentence with hyphenated-words.",
	pattern = "[:alpha:]+-?[:alpha:]+"
)


str_extract_all(
	string = "This isn't a sentence with hyphenated-words.",
	pattern = "[[:alpha:]']+-?[[:alpha:]']+"
)

str_extract_all(
	string = "This isn't a sentence with hyphenated-words.",
	pattern = "[[:alpha:]']+-?[[:alpha:]']+|[:alpha:]{1}"
)



##  ............................................................................
##  Stop words                                                              ####



length(stopwords(source = "smart"))
length(stopwords(source = "snowball"))
length(stopwords(source = "stopwords-iso"))


stopwords(language = "es", source = "snowball")



##  ............................................................................
##  Remover stop words                                                      ####

fir_tree <- hca_fairytales() %>%
	filter(book == "The fir tree",
			 language == "English")


tidy_fir_tree <- fir_tree %>%
	unnest_tokens(word, text)


# Usemos la lista de palabras vacías de Snowball como ejemplo. Dado que las
# palabras vacías regresan de esta función como un vector, usaremos filter().
 

tidy_fir_tree %>%
	filter(!(word %in% stopwords(source = "snowball")))



##  ............................................................................
##  Stemming                                                                ####


# cuando tenemos dos versiones de una palabra base a menudo se llama stem (raíz)

tidy_fir_tree <- fir_tree %>%
 unnest_tokens(word, text) %>%
 anti_join(get_stopwords())

# esta librería es para palabras parecidas
library(SnowballC)

tidy_fir_tree %>%
 mutate(stem = wordStem(word)) %>%
 count(stem, sort = TRUE)

# Lo alentamos a pensar en la lematización como un paso de preprocesamiento en
# el modelado de texto, uno que debe pensarse y elegirse (o no) con buen juicio.

# Stemming reduce el espacio de características de los datos de texto.

# Lematizar implica estandarizar, desambiguar, segmentar y, en caso de usar
# programas de lematización automática, también etiquetar.1​

# Hay otra opción para normalizar palabras a una raíz que adopta un enfoque
# diferente. En lugar de usar reglas para reducir las palabras a sus raíces, la
# lematización utiliza el conocimiento sobre la estructura de un idioma para
# reducir las palabras a sus lemas, las formas canónicas o de diccionario de las
# palabras.


##  ............................................................................
##  Incrustación de palabras                                                ####

 # algunos algoritmos pueden beneficiarse de las características de memoria de
 # las matrices dispersas (ej. regresión regularizada). Los algoritmos basados
 # en árboles no se desempeñan mejor con matrices dispersas (escasas).

 # el objetivo es reducir la cantidad de dimensiones que representan datos
 # de texto.

 # Las incrustaciones de palabras son una forma de representar datos de texto
 # como vectores de números basados en un gran corpus de texto, capturando el
 # significado semántico del contexto de las palabras.


 complaints <- read_csv("./09_redes_neuronales/complaints.csv.gz")

 # Vamos a crear una matriz dispersa, donde los elementos de la matriz son los
 # recuentos de palabras en cada documento.
 
quejas <- complaints |> 
 unnest_tokens(word, consumer_complaint_narrative) |> 
 anti_join(get_stopwords(), by = "word") %>%
 mutate(stem = wordStem(word)) %>%
 count(complaint_id, stem) %>%
 cast_dfm(complaint_id, stem, n)




# Las incrustaciones de palabras son una forma de representar datos de texto
# como vectores de números basados en un gran corpus de texto, capturando el
# significado semántico del contexto de las palabras.



##  ............................................................................
##  Scotus                                                                  ####

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












