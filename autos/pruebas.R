
import::from(magrittr, "%T>%", "%$%",  "%<>%", .into = "operadores")
import::from(parallel, detectCores, makePSOCKcluster, stopCluster)
import::from(doParallel, registerDoParallel)
import::from(janitor, make_clean_names)
pacman::p_load(finetune,
					embed,
					tictoc,
					neuralnet,
					parsnipExtra,
					textrecipes,
					tidymodels,
					tidyverse)

options(pillar.sigfig    = 5,
		  tibble.print_min = 10,
		  scipen = 999,
		  tidymodels.dark = TRUE,
		  digits = 7,
		  readr.show_col_types = FALSE,
		  dplyr.summarise.inform = FALSE)



# detener el backend
unregister <- function() {
	env <- foreach:::.foreachGlobals
	rm(list=ls(name=env), pos=env)
}

autos_raw <- ISLR::Auto |> as_tibble()
mk_00 <- c(`1` = "american",  `2` = "european", `3` = "japanese")
autos <- autos_raw |> 
	mutate(across(name, as.character)) |> 
	separate(name, c("make", "model", "trim"),
				extra = "drop", fill = "right") |> 
	mutate(
		across(trim, replace_na, "base"),
		across(origin, recode, !!!mk_00),
		across(where(is.character), as.factor),
		across(c(cylinders, year), as.factor))

ver <- . %>% prep() %>% juice()


autos_split <- initial_split(autos, prop = 0.8, strata = mpg)
autos_train <- training(autos_split)

autos_folds <- vfold_cv(autos_train, v = 10, strata = mpg)
autos_folds$splits[[1]] |> assessment()


uni_rec <- recipe(mpg ~ ., data = autos_train) |> 
	step_rm(trim) |>
	step_lencode_mixed(year, cylinders, outcome = vars(mpg)) |>
	step_YeoJohnson(all_numeric(), -all_nominal(), -all_outcomes()) |>
	step_dummy(origin, one_hot = T) |>
	step_corr(all_numeric_predictors(), threshold = 0.9) |> 
	step_impute_mode(model) |>
	step_other(make, model, threshold = 0.01) |>
	step_dummy_hash(make, model, signed = TRUE, num_terms = 16L) |>
	step_zv(all_predictors())

uni_rec |> ver()
uni_rec |> ver() |> select(!starts_with("dummy")) %>% sample_frac(0.5, replace = F) %>% print(n = 55)
uni_rec |> ver() |> select(starts_with("dummy")) %>% sample_frac(0.5, replace = F) %>% print(n = 25)

mlp_nnets <- mlp(hidden_units = tune(),
					  penalty      = 0.1,
					  epochs       = tune()) |>
 set_engine("nnet") |>
 set_mode("regression")

# show_engines("mlp") |> 
# 	filter(mode == "regression")


mlp_neural <- mlp(hidden_units = 2,
						penalty = 0.1,
						epochs  = 2) |>  
	set_engine("neuralnet") |> 
	set_mode("regression")  


# mlp_neural <- mlp(hidden_units = 2,
# 						# penalty = 0,
# 						epochs  = 2) |>  
# 	set_engine("neuralnet") |> 
# 	set_engine(engine    = "neuralnet",
# 				  algorithm = "rprop+",
# 				  err.fct   = "sse",
# 				  act.fct   = "logistic",
# 				  threshold = 0.1,
# 				  linear.output = TRUE) |>
# 	set_mode("regression")  


autos_workflow <- workflow_set(preproc = list(rec_all = uni_rec),
										 models  = list(nnet = mlp_nnets))

# metricas <- metric_set(rmse, rsq)

race_ctrl <- control_race(
 	save_pred     = TRUE, 
 	allow_par = FALSE,
 	parallel_over = "everything",
 	save_workflow = TRUE)


all_cores <- detectCores(logical = FALSE)
clusterpr <- makePSOCKcluster(all_cores)
registerDoParallel(clusterpr)

# 8.81 seg
# year + origin: 14.82
# year + origin + cylinder: 14.82
# + hash_make: 16.44
# + hast_model: 21.44
# all: 18.19
# all with penalty: 89.12
tic()
tune_res <- autos_workflow |> 
	workflow_map(
		"tune_race_anova",
		seed = 1503,
		resamples = autos_folds,
		verbose = TRUE,
		grid    = 20, 
		control = race_ctrl
		# metrics = metricas
		
		)
toc()

stopCluster(clusterpr)
unregister()


lm_wf <- workflow() |> 
	add_model(mlp_neural) |> 
	add_recipe(uni_rec)


neural_fit <- parsnip::fit(lm_wf, autos_train) |> 
	extract_fit_engine()
	


tune_res2 <- tune_res |> bind_rows(res_neural)

tune_res2 %>%
	rank_results(select_best = TRUE)
	filter(.metric == "rmse") |> 
	select(modelo = wflow_id, .config, rmse = mean, rank)


# https://bit.ly/3PGhtvs

neural_original <- neuralnet(mpg ~ ., data = uni_rec |> ver())


grid_ctrl <- control_grid(
	save_pred     = TRUE, 
	allow_par = FALSE
	parallel_over = "everything",
	save_workflow = TRUE)

mlp_param <- extract_parameter_set_dials(mlp_neural)

mmlp_param %>% extract_parameter_dials("penalty")

rg <- grid_regular(mlp_param, levels = 2)

tic()
tune_res <- autos_workflow %>% 
	workflow_map(
		"tune_grid", 
		verbose   = TRUE,
		resamples = autos_folds,
		control   = grid_ctrl,
		# param_info = updated_mlp_param,
		seed      = 2022,
		# metrics   = mset,
		grid      = crossing(
			hidden_units = 1:3,
			epochs = c(100, 200)
		))
toc()

res_neural <- fit(lm_wf, autos_train)

res <- lm_wf |> 
	fit_resamples(resamples = autos_folds, control = grid_ctrl)





grid_random(penalty(), size = 10)











tune_res %>%
	rank_results(select_best = TRUE) %>% 
	filter(.metric == "rmse") |> 
	select(modelo = wflow_id, .config, rmse = mean, rank)


best_results <- tune_res %>% 
	extract_workflow_set_result("rec_all_nnet") %>% 
	select_best(metric = "rmse")


tune_res %>% 
	extract_workflow("rec_all_nnet") %>% 
	finalize_workflow(best_results) %>% 
	last_fit(split = autos_split) |> 
	collect_metrics() |> filter(.metric == "rmse")





















































