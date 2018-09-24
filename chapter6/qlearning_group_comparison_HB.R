# メモリのクリア
rm(list=ls())
graphics.off()

# ライブラリの読み込み
library(tidyverse)
require(rstan)  

# 読み込むデータのsimulation ID
# generate_data_qlearning_group_comparison.Rで生成したファイルを読み込む
simulation_ID <- "Qlearning_group_comparison" 
csv_simulation_data <- paste0("./data/simulation_data", simulation_ID, ".csv")

# データの読み込み
data <- read.table(csv_simulation_data, header = T, sep = ",")

# データの情報
nSubject <- 40
nGroup <- 2
nTrial <- 100

# 以下のようにデータから抽出することも可能
# (グループごとの参加者の数が等しく，試行数も全参加で等しい場合)
# sublist <- dplyr::distinct(data, subject)$subject
# grouplist <- dplyr::distinct(data, group)
# nGroup <- length(grouplist)
# nSubject <- length(sublist) * nGroup
# nTrial <- length(filter(data,data$subject==1 & data$group==1)$trial)
  
# Stan用のデータリスト
dataList = list(  
  N = nSubject,
  G = rep(c(1,2), times = c(20,20)), # 参加者ごとのグループのインデックス
  nGroup = nGroup, 
  flgCommon_alpha = 1, # 1のとき，alphaの集団レベル分布はグループ間で共通
  flgCommon_beta = 1,  # 1のとき，alphaの集団レベル分布はグループ間で共通
  T = nTrial,
  c = matrix(data$choice, nSubject, nTrial, byrow = T),
  r = matrix(data$reward, nSubject, nTrial , byrow = T),
  WBICmode = 0         # 1のとき，WBICの計算のためのサンプリングをする
)

# -----------------------------------------------------------#
# Stanコードのコンパイル
#------------------------------------------------------------#
stanFit <- list()
stanFit_WBIC <- list()
smodels <- list()

# Stanコードのリスト (ここでは一つだけ)
modelfile <- c('model_qlearning_multiple_group.stan')

for (idxm in 1:length(modelfile)) {
  smodels[idxm] <- rstan::stan_model(file = paste0("./",modelfile[idxm]))
}

nStanModel <- length(smodels)

# -----------------------------------------------------------#
# Set models
#------------------------------------------------------------#
# モデル数
# flgCommon_alpha, flgCommon_betaの違うモデルは一つとカウントする
nModel <- 2 
idxStanModel <- c(1,1) # それぞれに対応するStanコードのインデックス

setDataForEachModel <- function(dataList, idxm) {
  if (idxm == 1) {
    # Model 1: 学習率，逆温度ともに集団分布は共有
    dataList$flgCommon_alpha = 1
    dataList$flgCommon_beta = 1
  } else if (idxm == 2) {
    # Model 2: 学習率は集団レベル分布平均は異なる，
    # 逆温度は集団分布は共有
    dataList$flgCommon_alpha = 0
    dataList$flgCommon_beta = 1
  } else if (idxm == 3) {
    # Model 3: 学習率は集団分布は共有，
    # 逆温度は集団レベル分布平均は異なる
    dataList$flgCommon_alpha = 1
    dataList$flgCommon_beta = 0
  } else if (idxm == 4) {
    # Model 4: 学習率，逆温度ともに集団レベル分布平均は異なる
    dataList$flgCommon_alpha = 0
    dataList$flgCommon_beta = 0
  } else {
    print("Wrong model index for setDataForEachModel. (idxm must be 1 <= idxm <=4)")
  }
  return(dataList)
}

#------------------------------------------------------------------------#
# RUN THE MCMC
#------------------------------------------------------------------------#
stanFit <- list()
stanFit_WBIC <- list()

# 以下を実行すると並列化できる
# rstan_options(auto_write=TRUE)
# options(mc.cores=parallel::detectCores())

nChains <- 3

for (idxm in 1:nModel) { 
  # Get MC sample of posterior:
  # hierarchical bayesian model
  dataList$WBICmode = 0
  
  # # initize chains from MLE estimates -------------------- # 
  initsList <- vector("list",nChains)
  for (idxChain in 1:nChains) {
    initsList[[idxChain]] <- list(
      alpha = runif(nSubject,0.3,0.6),
      beta = runif(nSubject,0.5,2.0),
      mu_p_alpha = runif(nGroup,-0.1,0.1),
      sigma_p_alpha = runif(nGroup,0.5,1),
      mu_p_beta = runif(nGroup,-3,-2),
      sigma_p_beta = runif(nGroup,0.4,0.9), 
      eta_alpha = runif(nSubject,-0.2,0.2),
      eta_beta = runif(nSubject,-0.2,0.2)
    )
  }
  
  dataList$WBICmode = 0
  
  stanFit[idxm] <- rstan::sampling( object=smodels[[idxStanModel[idxm]]] , 
                             data = setDataForEachModel(dataList,idxm), 
                             chains = nChains,
                             pars = c('mu_p_alpha',
                                      'sigma_p_alpha',
                                      'mu_p_beta',
                                      'sigma_p_beta',
                                      'alpha_p',
                                      'beta_p',
                                      'log_lik',
                                      'alpha_diff', 'beta_diff'),
                             iter = 5000,
                             warmup = 1000,  
                             thin = 1,
                             init = initsList 
  )
}


# 集団レベル分布平均の事後分布のプロット -------------------------
x11()
rstan::stan_plot(stanFit[[2]],
                      point_est="mean",
                      show_density="T",
                      ci_level = 0.95,
                      pars = c("alpha_p",'alpha_diff'))


x11()
g_model2 <- rstan::stan_plot(stanFit[[2]],
                             point_est="mean",
                             show_density="T",
                             fill_color="gray55", 
                             ci_level = 0.95,
                             c('alpha_p',
                               'alpha_diff'
                             ))
print(g_model2)
print(g_model2$data,digits = 2)

# 図を保存する場合
# ggsave(file="./figs/group_comparison_posterior.eps", g_model2)

# x11()
# traceplot(stanFit[[2]],pars=c("mu_p_alpha","lp__"))

# WAICの計算 ----------------------------------------
waic <- array()
lppd <- array()
p_waic <- array()
for (idxm in 1:nModel) {
  log_lik <- rstan::extract(stanFit[[idxm]],"log_lik")$log_lik

  lppd[idxm] <- mean(log(colMeans(exp(log_lik))))
  p_waic[idxm] <- mean(colMeans(log_lik^2) - colMeans(log_lik)^2)
  
  waic[idxm] <- - 2 * lppd[idxm] + 2* p_waic[idxm]
}

wbic <- array()

# WBICの計算 ---------------------------------------
for (idxm in 1:nModel) { 
  
  dataList$WBICmode = 1
  
  stanFit_WBIC[idxm] <- rstan::sampling( object=smodels[[idxStanModel[idxm]]] , 
                             data = setDataForEachModel(dataList,idxm), 
                             chains = nChains ,
                             pars = c('log_lik'),
                             iter = 5000,
                             warmup = 1000, 
                             thin = 1, 
                             init = initsList 
  )
  
  log_lik <- rstan::extract(stanFit_WBIC[[idxm]],"log_lik")$log_lik
  wbic[idxm] <- - mean(rowSums(log_lik))
}

#------------------------------------------------------#
# Bayes factor
#------------------------------------------------------#
mcomp <- list(c(2,1), c(1,2)) # , c(2,3), c(2,4))

for (idx in 1:length(mcomp)) {
  m1 <- mcomp[[idx]][1]
  m2 <- mcomp[[idx]][2]
  
  cat("[Bayes factor (WBIC)]    model ", m1, 
      "over model ", m2, ": ", 
      exp( -wbic[m1] - (-wbic[m2])), 
      "\n")
}
