pacman::p_load(tictoc, neuralnet, tidyverse)


tic()
neural_res <- neuralnet(mpg ~ ., 
								data = train_data,
								hidden = 10,
								stepmax = 1e7,
								algorithm = "rprop+",
								err.fct = "sse",
								act.fct = "logistic",
								threshold = 0.1,
								linear.output = TRUE)
toc()


