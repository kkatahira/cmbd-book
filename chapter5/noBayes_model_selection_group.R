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
source("parameter_fit_functions_group.R")

# 読み込むデータのsimulation ID (モデルフィッティングの結果の保存にも使用する)
simulation_ID <- "FQlearning_group"

# 読み込むデータファイルの指定
csv_simulation_data <- paste0("./data/simulation_data", simulation_ID, ".csv")


#----------------------------------------------------------#
# モデルフィッティング
#----------------------------------------------------------#

# データファイルの読み込み---------------
data <- read.table(csv_simulation_data, header = T, sep = ",")

sublist <- dplyr::distinct(data, subject)$subject
nSubject <- length(sublist)
nTrial <- sum(data$subject==1)

# 個人レベル最尤推定 -----------------------------
cat("----------- single subject  ML -----------\n")
resultSSML <- paramfitSSML(modelfunctions, data, nParamList)

# 固定効果最尤推定 -------------------------------
cat("----------- fixed effect ML -----------\n")
# resultFEML <- paramfitFEML(modelfunctions, data, nParamList)

# MAP推定 ----------------------------------------
cat("----------- MAP -----------\n")
resultMAP <- paramfitSSMAP(modelfunctions, data, nParamList, priorList)

# plot posterior
# rstan::stan_hist(stanFit[[1]])


# ランダム効果モデル選択法 (要Matlab)------------------
# # 書き出し
# write.table(dfSSMAP_LML, file = "./results/SSMAP_LML.csv",
#            quote = FALSE, sep = ",",row.names = FALSE)
# 
# # 以下をSPMをロードしたMatlab上で実行する
# lme = csvread('./results/SSMAP_LML.csv');
# [alpha,exp_r,xp,pxp,bor] = spm_BMS(lme);

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



csv_results <- paste0("./results/model_selection_", simulation_ID, ".csv")


#------------------------------------------------------#
# モデル間のAICのt検定
#------------------------------------------------------#
# Model 1 (Q学習) vs Model 2 (F-Q学習)
t.test(dfSSML_AIC$value[dfSSML_AIC$model=="m1"], 
       dfSSML_AIC$value[dfSSML_AIC$model=="m2"], paired = TRUE)

# Model 2 (F-Q学習) vs # Model 3 (DF-Q学習)
t.test(dfSSML_AIC$value[dfSSML_AIC$model=="m2"], 
       dfSSML_AIC$value[dfSSML_AIC$model=="m3"], paired = TRUE)

if (0){
#------------------------------------------------------#
# 固定効果モデルに基づく尤度比検定
#------------------------------------------------------#
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
}

#------------------------------------------------------#
# 個人ごとの尤度比検定
#------------------------------------------------------#
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


# ランダム効果モデル選択法 (要Matlab)------------------

dfSSMAP_LML <- data.frame(subject = 1:nSubject, 
                          m1 = resultMAP$lml[[1]], 
                          m2 = resultMAP$lml[[2]], 
                          m3 = resultMAP$lml[[3]])

# # 書き出し
write.table(dfSSMAP_LML[c("m1","m2","m3")], 
            file = "./results/SSMAP_LML.csv",
            quote = FALSE, sep = ",",row.names = FALSE,col.names = FALSE)
# 
# # 以下をSPMをロードしたMatlab上で実行する
# lme = csvread('./results/SSMAP_LML.csv');
# [alpha,exp_r,xp,pxp,bor] = spm_BMS(lme);

#------------------------------------------------------#
# 集団全体に対する，個人レベルモデルに基づく尤度比検定
#------------------------------------------------------#
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
