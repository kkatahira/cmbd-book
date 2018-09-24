#----------------------------------------------------------#
# 階層ベイズ法を用いた特性との相関分析
#----------------------------------------------------------#

# メモリのクリア
rm(list=ls())
graphics.off()

# ライブラリの読み込み
library(tidyverse)
require(rstan)  

# 乱数のシードを設定
set.seed(141)

# 読み込むデータのsimulation ID
simulation_ID <- "Qlearning_trait_correlation_random_slope" 
csv_simulation_data <- paste0("./data/simulation_data", simulation_ID, ".csv")

data <- read.table(csv_simulation_data, header = T, sep = ",")

csv_param <- paste0("./data/trueparam_",simulation_ID, ".csv")
data_param <- read.table(csv_param, header = T, sep = ",")

sublist <- unique(data$subject)
nSubject <- length(sublist)     # 参加者数
nTrial <- data %>% filter(subject == sublist[1]) %>% nrow()      # 参加者ごとの試行数 (今は全参加者で試行数が等しいとする)

# Stan用の設定 ----------------------------------------------------------------

# Stan用のデータリスト
dataList = list(   
  N = nSubject,
  T = nTrial,
  c = matrix(data$choice, nSubject, nTrial, byrow = T),
  r = matrix(data$reward, nSubject, nTrial , byrow = T),
  trait = data_param$trait,
  flg_trait_alpha = 1,
  WBICmode = 0
)

stanFit <- list()

# model_list
modelfile <- c('smodel_qlearning_trait.stan', # fixed effect slope
               'smodel_qlearning_trait_random_slope.stan' # random effect slope
)

model_path <- "./"
smodels <- list()

# Stanコードのコンパイル
# compile models
for (idxm in 1:length(modelfile)) {
  smodels[idxm] <- rstan::stan_model(file = paste0(model_path,modelfile[idxm]))
}

nModel <- 2 # total number of models
idxStanModel <- c(1,2)

setDataForEachModel <- function(dataList, idxm) {
  if (idxm == 1) {
    dataList$flg_trait_alpha = 1
  } else if (idxm == 2) {
    dataList$flg_trait_alpha = 1
  } else if (idxm == 3) {
    dataList$flg_trait_alpha = 0
  } else {
    print("Wrong model index for setDataForEachModel. (idxm must be 1 <= idxm <=3)")
  }
  return(dataList)
}


# MCMCサンプリングの実行 -----------------------------------------------------------

# 以下で並列化
# rstan_options(auto_write=TRUE)
# options(mc.cores=parallel::detectCores())

nChains <- 3

stanFit <- list()

for (idxm in 1:nModel) {
  
  dataList$WBICmode = 0
  
  initsList <- vector("list",3)
  if (idxStanModel[idxm] == 1) {
    # Model 1の設定
    
    # 初期値
    for (idxChain in 1:nChains) {
      initsList[[idxChain]] <- list(
        mu_p_alpha = runif(1,-0.1,0.1),
        sigma_p_alpha = runif(1,0.5,1),
        mu_p_beta = runif(1,-3,-2),
        sigma_p_beta = runif(1,0.4,0.9),
        eta_alpha = runif(nSubject,-0.2,0.2),
        eta_beta = runif(nSubject,-0.2,0.2),
        b1 = runif(1,-0.5,0.5)
      )
    }
    
    # MCMCを記録するパラメータ
    parslist <- c(
      'mu_p_alpha',
      'sigma_p_alpha',
      'mu_p_beta',
      'sigma_p_beta',
      'alpha_p',
      'beta_p',
      'b1',
      'log_lik'
    )
  } else {
    # Model 2の設定
    
    # 初期値
    for (idxChain in 1:nChains) {
      initsList[[idxChain]] <- list(
        mu_p_alpha0 = runif(1,-0.1,0.1),
        sigma_p_alpha0 = runif(1,0.5,1),
        mu_p_alpha1 = runif(1,-0.1,0.1),
        sigma_p_alpha1 = runif(1,0.5,1),
        mu_p_beta = runif(1,-3,-2),
        sigma_p_beta = runif(1,0.4,0.9),
        eta_alpha0 = runif(nSubject,-0.2,0.2),
        eta_alpha1 = runif(nSubject,-0.2,0.2),
        eta_beta = runif(nSubject,-0.2,0.2)
      )
    }
    
    # MCMCを記録するパラメータ
    parslist <- c(
      'mu_p_alpha1',
      'sigma_p_alpha1',
      'mu_p_alpha0',
      'sigma_p_alpha0',
      'mu_p_beta',
      'sigma_p_beta',
      'alpha_p',
      'beta_p',
      'b0',
      'b1',
      'log_lik'
    )
  }
  cat("Sampling model", idxm, "(stanmodels :", idxStanModel[idxm],")...\n")
  
  dataList$WBICmode = 0
  
  stanFit[idxm] <- rstan::sampling( object=smodels[[idxStanModel[idxm]]] ,
                                    data = setDataForEachModel(dataList,idxm),
                                    chains = nChains ,
                                    pars = parslist,
                                    iter = 5000,
                                    warmup = 1000,
                                    thin = 1,
                                    init = initsList
  )
  
}

# Model 1 (fixed effect slope)の結果 -----------------------------------------

# plot posterior
a <- rstan::stan_plot(stanFit[[1]],
                      point_est="mean",
                      show_density="T",
                      ci_level = 0.95,
                      pars = c("b1"))
b <- rstan::extract(stanFit[[1]],"b1")$b1

# b1の事後分布の要約
print(a$data,digits = 3)

ggplot() + theme_set(theme_bw(base_size = 18)) 

x11()
g <- ggplot(data=data.frame(value=b), aes(x=value)) +
  geom_histogram(aes(y = ..density..)) +
  geom_density(size=1,linetype=1) +
  geom_line(data=data.frame(x=c(a$data$l, a$data$h)),
            aes(x=x,y=-0.02), size=3) +
  labs(title="Fixed effect slope") +
  ylab('density') +
  xlab('b1')

print(g)


# Model 2 (random effect slope)の結果 ---------------------------------------

# plot posterior
a <- rstan::stan_plot(stanFit[[2]],
                      point_est="mean",
                      show_density="T",
                      ci_level = 0.95,
                      pars = c("mu_p_alpha1"))

b <- rstan::extract(stanFit[[2]],"mu_p_alpha1")$mu_p_alpha1

# mu_p_alpha1の事後分布の要約
print(a$data,digits = 3)

x11()
g <- ggplot(data=data.frame(value=b), aes(x=value)) +
  geom_histogram(aes(y = ..density..)) +
  geom_density(size=1,linetype=1) +
  geom_line(data=data.frame(x=c(a$data$l, a$data$h)),
            aes(x=x,y=-0.02), size=3) +
  labs(title="Random effect slope (mean)") +
  ylab('density') +
  xlab('mu_p_alpha1')

print(g)
