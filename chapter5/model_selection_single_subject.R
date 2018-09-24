# -------------------------------------------------------- #
# 参加者一人分のデータからパラメータ推定・モデル選択を行う。
# -------------------------------------------------------- #

# メモリのクリア，図を閉じる
rm(list=ls())
graphics.off()

# ライブラリの読み込み
library(tidyverse)
require(rstan)  

# 乱数のシードを設定
set.seed(141)

# モデル関数の読み込み
source("model_functions_FQ.R")

# パラメータ推定用関数の読み込み
source("parameter_fit_functions.R")

# 読み込むデータのsimulation ID (モデルフィッティングの結果の保存にも使用する)
simulation_ID <- "FQlearning_siglesubject" 

# 読み込むデータファイルの指定
csv_simulation_data <- paste0("./data/simulation_data", simulation_ID, ".csv")

# データファイルの読み込み
dt <- read.table(csv_simulation_data, header = T, sep = ",")
data <- list(reward = dt$r, choice = dt$choice) 

# パラメータ推定 --------------------------------------------------------------

# 最尤推定 
cat("----------- ML -----------\n")
resultML <- paramfitML(modelfunctions, data, nParamList)

# MAP推定 
cat("----------- MAP -----------\n")
resultMAP <- paramfitMAP(modelfunctions, data, nParamList, priorList)

# ベイズ推定 
cat("----------- Bayes -----------\n")

# サンプリングを並列化する場合は以下を実行
# rstan_options(auto_write=TRUE)
# options(mc.cores=parallel::detectCores())
  
# Stan用データのリスト作成
dataList = list(   
  c = as.vector(data$choice),
  r = as.vector(data$reward),
  T = length(data$reward)
)

stanFit <- list()
stanFit_WBIC <- list()

# モデルのStanコードのリスト　順にmodel 1, model2, model 3として扱う
modelfile <- c('smodel_qlearning_single.stan',
               'smodel_fqlearning_single.stan',
               'smodel_dfqlearning_single.stan')
modelfileWBIC <- c('smodel_qlearning_single_WBIC.stan',
               'smodel_fqlearning_single_WBIC.stan',
               'smodel_dfqlearning_single_WBIC.stan')

# サンプリング
for (idxm in 1:nModel) {
  stanFit[idxm] <- stan(file=modelfile[idxm], 
                        data=dataList, iter=5000, thin=1, chains=3)
}

# 事後分布のプロット
rstan::stan_hist(stanFit[[1]])

# WAICの計算 -----------------------------------------------------------------

waic <- array()
lppd <- array()
p_waic <- array()
for (idxm in 1:nModel) {
  log_lik <- rstan::extract(stanFit[[idxm]],"log_lik")$log_lik
  
  lppd[idxm] <- mean(log(colMeans(exp(log_lik))))
  p_waic[idxm] <- mean(colMeans(log_lik^2) - colMeans(log_lik)^2)
  
  waic[idxm] <- - 2 * lppd[idxm] + 2* p_waic[idxm]
}

# WBICの計算 -----------------------------------------------------------------

wbic <- array()
for (idxm in 1:nModel) {
  stanFit_WBIC[idxm] <- stan(file=modelfileWBIC[idxm], 
                             data=dataList, iter=5000, thin=1, chains=3)
  
  log_lik <- rstan::extract(stanFit_WBIC[[idxm]],"log_lik")$log_lik
  wbic[idxm] <- - mean(rowSums(log_lik))
}

#　結果をデータフレームに保存
dfmodels <- data.frame(model = c("qlearning", "fqlearing","dfqlearning"), 
                       nParam = nParamList, 
                       ll = - resultML$negll, 
                       aic = - resultML$aic/2, 
                       waic = - waic * dataList$T /2,
                       lml_Laplace = resultMAP$lml, 
                       bic = - resultML$bic/2, 
                       wbic = - wbic
)

csv_results <- paste0("./results/model_selection_", simulation_ID, ".csv")

# 結果を書き出す
write.table(dfmodels, file = csv_results, 
            quote = FALSE, sep = ",",row.names = FALSE)


# 尤度比検定 -------------------------------------------------------------------

# 比較するペアのリスト (対立モデル，ヌルモデル)
mcomp <- list(c(3,2), c(3,1))

for (idx in 1:length(mcomp)) {
  
  # 検定統計量 (2×対数尤度の差分)
  D <- 2*(dfmodels$ll[mcomp[[idx]][1]] - dfmodels$ll[mcomp[[idx]][2]]) 
  
  # 自由度
  df <- dfmodels$nParam[mcomp[[idx]][1]] - dfmodels$nParam[mcomp[[idx]][2]] 
  
  # p値
  p <- 1 - pchisq(D,df) 
  
  cat("alternative model ", mcomp[[idx]][2], 
      ", null model:", mcomp[[idx]][1], 
      ", D:", D, ", df:", df, ", p-value:", p, "\n")
}

# ベイズファクターの計算 -------------------------------------------------------------

# 比較するペアのリスト (対立モデル，ヌルモデル)
mcomp <- list(c(3,2), c(3,1), c(2,3))

for (idx in 1:length(mcomp)) {
  m1 <- mcomp[[idx]][1]
  m2 <- mcomp[[idx]][2]
  
  cat("[Bayes factor (WBIC)]    model ", m1, 
      "over model ", m2, ": ", 
      exp( dfmodels$wbic[m1] - dfmodels$wbic[m2]), 
      "\n")
  cat("[Bayes factor (Laplace)] model ", m1, 
      "over model ", m2, ": ", 
      exp( dfmodels$lml_Laplace[m1] - dfmodels$lml_Laplace[m2]), 
      "\n")
  cat("[Bayes factor (BIC)]     model ", m1, 
      "over model ", m2, ": ", 
      exp( dfmodels$bic[m1] - dfmodels$bic[m2]), 
      "\n")
}
