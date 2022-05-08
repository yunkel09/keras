#   ____________________________________________________________________________
#   Text Analsys                                                            ####


# Para crear variables para el aprendizaje automático supervisado a partir del
# lenguaje natural, necesitamos alguna forma de representar el texto sin
# procesar como números para que podamos realizar cálculos sobre ellos.

pacman::p_load(tokenizer, tidytext, hcandersenr, tidyverse)


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














































