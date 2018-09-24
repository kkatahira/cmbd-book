#------------------------------------------------------------#
# 集団データに対し，各種パラメータ推定法と各種モデル選択を行う
#------------------------------------------------------------#

# メモリのクリア，図を閉じる
rm(list=ls())
graphics.off()

# ライブラリの読み込み
library(tidyverse)
require(rstan)  

# 乱数のシードを設定
set.seed(1)

# モデル関数の読み込み
source("model_functions_FQ.R")

# パラメータ推定用関数の読み込み
source("parameter_fit_functions_group.R")

# 読み込むデータのsimulation ID (モデルフィッティングの結果の保存にも使用する)
simulation_ID <- "FQlearning_group"

# 読み込むデータファイルの指定
csv_simulation_data <- paste0("./data/simulation_data", simulation_ID, ".csv")


# モデルフィッティング --------------------------------------------------------------

# データファイルの読み込み
data <- read.table(csv_simulation_data, header = T, sep = ",")

sublist <- dplyr::distinct(data, subject)$subject
nSubject <- length(sublist)
nTrial <- sum(data$subject==1)

# 個人レベル最尤推定
cat("----------- single subject  ML -----------\n")
resultSSML <- paramfitSSML(modelfunctions, data, nParamList)

# 固定効果最尤推定
cat("----------- fixed effect ML -----------\n")
resultFEML <- paramfitFEML(modelfunctions, data, nParamList)

# 個人レベルMAP推定
cat("----------- MAP -----------\n")
resultMAP <- paramfitSSMAP(modelfunctions, data, nParamList, priorList)

# ベイズ推定 (階層ベイズ法)
cat("----------- Bayes -----------\n")

# サンプリングを並列化する場合は以下を実行
# rstan_options(auto_write=TRUE)
# options(mc.cores=parallel::detectCores())

# Stan用データのリスト作成
dataList = list(    
  # 選択cと報酬rは参加者ID×試行×行列にする
  # この方法は参加者によって試行数が異なる場合は使えない
  c = matrix(data$choice, nSubject, nTrial, byrow = T),
  r = matrix(data$reward, nSubject, nTrial , byrow = T),
  N = nSubject,
  T = nTrial,
  WBICmode = 0 # WBICを計算するときは1にする
)

stanFit <- list()
stanFit_WBIC <- list()

# モデルのStanコードのリスト　順にmodel 1, model2, model 3として扱う
# WBICを計算する場合は変数WBICmodeで切り替えるようにするため，Stanコードは3つのみ
smodels <- c('smodel_qlearning_group.stan',
               'smodel_fqlearning_group.stan',
               'smodel_dfqlearning_group.stan')

nModel <- length(smodels)

nChains <- 3

# モデルごとに...
for (idxm in 1:nModel) { 

  # MCMCの初期値を個人レベルの最尤推定値に近くなるよう設定する
  initsList <- vector("list",3)
  if (idxm == 3) {
    for (idxChain in 1:nChains) {
      initsList[[idxChain]] <- list(
        alpha = runif(nSubject,0.1,0.9),
        alphaF = runif(nSubject,0.1,0.9),
        beta = runif(nSubject,0.5,2.0),
        mu_p_alpha = runif(1,0.1,0.9),
        mu_p_alphaF = runif(1,0.1,0.9),
        sigma_p_alpha = runif(1,0.1,0.5),
        sigma_p_alphaF = runif(1,0.1,0.5),
        mu_p_beta = runif(1,0.1,0.9),
        sigma_p_beta = runif(1,0.1,0.9)
      )
    }
  } else {
    for (idxChain in 1:nChains) {
      initsList[[idxChain]] <- list(
        alpha = runif(nSubject,0.1,0.9),
        beta = runif(nSubject,0.5,2.0),
        mu_p_alpha = runif(1,0.1,0.9),
        sigma_p_alpha = runif(1,0.1,0.5),
        mu_p_beta = runif(1,0.1,0.9),
        sigma_p_beta = runif(1,0.1,0.9)
      )
    }
  }
  
  # run MCMC
  stanFit[idxm] <- stan(file = smodels[[idxm]], 
                  data = dataList, 
                  iter = 5000, 
                  thin = 1, 
                  chains = nChains, 
                  warmup = 1000,
                  init = initsList)
  
}

# plot posterior
# rstan::stan_hist(stanFit[[1]])

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

# WBICの計算 ----------------------------------------
wbic <- array()

for (idxm in 1:nModel) { 
  dataList$WBICmode = 1
  
  stanFit_WBIC[idxm] <- stan(file = smodels[[idxm]], 
                             data = dataList, 
                             iter = 5000, 
                             thin = 1, 
                             chains = nChains, 
                             warmup = 1000,
                             init = initsList)

  log_lik <- rstan::extract(stanFit_WBIC[[idxm]],"log_lik")$log_lik
  wbic[idxm] <- - mean(rowSums(log_lik))
}


# 対数尤度とAICのプロット------------------------------

# results of SSML, SSMAP
dfSSMAP_LML <- data.frame(subject = 1:nSubject, 
                          m1 = resultMAP$lml[[1]], 
                          m2 = resultMAP$lml[[2]], 
                          m3 = resultMAP$lml[[3]])
dfSSMAP_LML <- tidyr::gather(dfSSMAP_LML,model,value,-subject)
dfSSMAP_LML$criteria <- "LML"

# results of SSML, LL
dfSS_LL <- data.frame(subject = 1:nSubject, 
                          m1 = -resultSSML$negll[[1]], 
                          m2 = -resultSSML$negll[[2]], 
                          m3 = -resultSSML$negll[[3]])
dfSSML_LL <- tidyr::gather(dfSS_LL,model,value,-subject)
dfSSML_LL$criteria <- "LL"

# results of SSML, AIC
dfSSML_AIC <- data.frame(subject = 1:nSubject, 
                      m1 = -resultSSML$aic[[1]]/2, 
                      m2 = -resultSSML$aic[[2]]/2, 
                      m3 = -resultSSML$aic[[3]]/2) %>% 
  tidyr::gather(model,value,-subject)
dfSSML_AIC$criteria <- "-AIC/2"
tmp <- rbind(dfSSML_LL, dfSSML_AIC)

tmp$criteria <- factor(tmp$criteria, levels = c("LL", "-AIC/2"))
  
ggplot()+theme_set(theme_bw(base_size = 14))
x11()
g <- ggplot(tmp, aes(x=model,y=value, group=subject)) + 
  geom_line() +
  geom_point() + facet_wrap(~criteria) +
  theme(axis.text.x = element_text(size=20),axis.text.y = element_text(size=20)) +
  xlab("Model") +
  ylab("") +
  scale_x_discrete(breaks=c('m1','m2','m3'),
                   labels = c('Q', 'F-Q','DF-Q')) 
print(g)

# 各種指標の保存---------------------------------------
dfmodels <- data.frame(model = c("qlearning", "fqlearing","dfqlearning"), 
                       nParam = nParamList, 
                       ll = - unlist(resultFEML$negll), 
                       aic = - unlist(resultFEML$aic)/2,  
                       waic = - waic * dataList$T * dataList$N /2,
                       bic = - unlist(resultFEML$bic)/2, 
                       wbic = - wbic
)

csv_results <- paste0("./results/model_selection_", simulation_ID, ".csv")

# write results to csv file
write.table(dfmodels, file = csv_results, 
            quote = FALSE, sep = ",",row.names = FALSE)


# モデル間のAICのt検定 ------------------------------------------------------------

# 標準的なQ学習 vs. F-Q学習
t.test(dfSSML_AIC$value[dfSSML_AIC$model=="m1"], 
       dfSSML_AIC$value[dfSSML_AIC$model=="m2"], paired = TRUE)

# F-Q学習 vs. DF-Q学習 
t.test(dfSSML_AIC$value[dfSSML_AIC$model=="m2"], 
       dfSSML_AIC$value[dfSSML_AIC$model=="m3"], paired = TRUE)


# 固定効果モデルに基づく尤度比検定 --------------------------------------------------------

mcomp <- list(c(3,2), c(3,1))

ll <- c(-resultFEML$negll[[1]], 
        -resultFEML$negll[[2]], 
        -resultFEML$negll[[3]]) 

for (idx in 1:length(mcomp)) {
  
  # difference of the loglikelihood
  D <- 2*(ll[mcomp[[idx]][1]] - ll[mcomp[[idx]][2]]) 
  
  # difference of the numbers of free parameters
  df <- nParamList[mcomp[[idx]][1]] - nParamList[mcomp[[idx]][2]] 
  
  # p-value 
  p <- 1 - pchisq(D,df) 
  
  cat("alternative model ", mcomp[[idx]][1], 
      ", null model:", mcomp[[idx]][2], 
      ", D:", D, ", df:", df, ", p-value:", p, "\n")
}


# 参加者ごとの尤度比検定 --------------------------------------------------------------

mcomp <- list(c(3,2), c(3,1)) # (alternative, null)

options(nsmall=3)

for (idxSub in 1:nSubject) {
  cat("Subject", idxSub, "\n")
  
  for (idxModel in 1:length(mcomp)) {
    
    # difference of the loglikelihood
    D <- 2*(-resultSSML$negll[[mcomp[[idxModel]][1]]][idxSub] - 
              (-resultSSML$negll[[mcomp[[idxModel]][2]]][idxSub])
            ) 
    
    # difference of the numbers of free parameters
    df <- nParamList[mcomp[[idxModel]][1]] - nParamList[mcomp[[idxModel]][2]] 
    
    # p-value 
    p <- 1 - pchisq(D,df) 
    
    cat("alternative model ", mcomp[[idxModel]][1], 
        ", null model:", mcomp[[idxModel]][2], 
        ", D:", format(D, nsmall = 2), ", df:", df, 
        ", p-value:", format(p, nsmall = 2), "\n")
  }
}



# 集団全体に対する，個人レベルモデルに基づく尤度比検定 ----------------------------------------------

mcomp <- list(c(3,2), c(3,1)) # (alternative, null)

for (idxModel in 1:length(mcomp)) {
  
  # difference of the loglikelihood
  D <- 2*(-sum(resultSSML$negll[[mcomp[[idxModel]][1]]]) - 
            (-sum(resultSSML$negll[[mcomp[[idxModel]][2]]]))
  ) 
  
  cat("\nsum ll (alt):",
      - sum(resultSSML$negll[[mcomp[[idxModel]][1]]]),
      "sum ll (null)",
      - sum(resultSSML$negll[[mcomp[[idxModel]][2]]]), "\n")
  
  # difference of the numbers of free parameters
  df <- (nParamList[mcomp[[idxModel]][1]] - nParamList[mcomp[[idxModel]][2]]) * nSubject
  
  # p-value 
  p <- 1 - pchisq(D,df) 
  
  cat("alternative model ", mcomp[[idxModel]][1], 
      ", null model:", mcomp[[idxModel]][2], 
      ", D:", format(D, nsmall = 2), ", df:", df, 
      ", p-value:", format(p, nsmall = 2), "\n")
}

# ベイズファクター ----------------------------------------------------------------

mcomp <- list(c(3,2), c(2,1), c(3,1), c(2,3))

for (idx in 1:length(mcomp)) {
  m1 <- mcomp[[idx]][1]
  m2 <- mcomp[[idx]][2]
  
  cat("[Bayes factor (WBIC)]    model ", m1, 
      "over model ", m2, ": ", 
      exp( dfmodels$wbic[m1] - dfmodels$wbic[m2]), 
      "\n")
}


# ランダム効果モデル選択法 (要Matlab)---------------------------------------
# # 書き出し
write.table(dfSSMAP_LML[c("m1","m2","m3")], 
            file = "./results/SSMAP_LML.csv",
            quote = FALSE, sep = ",",row.names = FALSE,col.names = FALSE)
# 
# # 以下をSPMをロードしたMatlab上で実行する
# lme = csvread('./results/SSMAP_LML.csv');
# [alpha,exp_r,xp,pxp,bor] = spm_BMS(lme);
# xpに超過確率が入る

