#----------------------------------------------------------#
# パラメータの点推定値 (最尤推定，MAP推定) を用いる群間比較,
# 固定効果モデルのモデル選択による群間比較
#----------------------------------------------------------#

# メモリのクリア
rm(list=ls())
graphics.off()

# ライブラリの読み込み
library(tidyverse) 

# 乱数のシードの設定
set.seed(141)

# モデル関数の読み込み
source("model_functions_Q_mulitiple_group.R")

# パラメータ推定用関数の読み込み
source("parameter_fit_functions_multiple_group.R")

# 読み込むデータのsimulation ID
simulation_ID <- "Qlearning_group_comparison" 
csv_simulation_data <- paste0("./data/simulation_data", simulation_ID, ".csv")

data <- read.table(csv_simulation_data, header = T, sep = ",")


# モデルフィッティング (個人レベル)，差の検定 -------------------------------------------------

# single subject ML
resultSSML <- paramfitSSML(modelfunctions[1], data, nParamList = 2)

# t-test (learning rate)
x1 <- resultSSML$param.1[resultSSML$group==1]
x2 <- resultSSML$param.1[resultSSML$group==2]
t.test(x1,x2,paired = F, var.equal = T)

# t-test (beta)
x1 <- resultSSML$param.2[resultSSML$group==1]
x2 <- resultSSML$param.2[resultSSML$group==2]
t.test(x1,x2,paired = F, var.equal = T)

# MAP 
resultSSMAP <- paramfitSSMAP(modelfunctions[1], data, nParamList = 2, priorList)

# t-test (learning rate)
x1 <- resultSSMAP$param.1[resultSSMAP$group==1]
x2 <- resultSSMAP$param.1[resultSSMAP$group==2]
t.test(x1,x2,paired = F, var.equal = T)

# t-test (beta)
x1 <- resultSSMAP$param.2[resultSSMAP$group==1]
x2 <- resultSSMAP$param.2[resultSSMAP$group==2]
t.test(x1,x2,paired = F, var.equal = T)


# モデルフィッティング (固定効果)，尤度比検定 -------------------------------------------------

# fixed effect ML
resultFEML <- paramfitFEML(modelfunctions, gpmap, data)

# 尤度比検定 
# model 1: 学習率はグループ間で共通
# model 2: 学習率はグループ間で異なる
mcomp <- list(c(2,1)) 

# 尤度の配列にする
ll <- c(-resultFEML$negll[[1]], 
        -resultFEML$negll[[2]]
) 

for (idx in 1:length(mcomp)) {
  
  # 検定統計量
  D <- 2 * (ll[mcomp[[idx]][1]] - ll[mcomp[[idx]][2]]) 
  
  # 自由度
  df <- nParamList[mcomp[[idx]][1]] - nParamList[mcomp[[idx]][2]] 
  
  # p値
  p <- 1 - pchisq(D,df) 
  
  cat("alternative model ", mcomp[[idx]][1], 
      ", null model:", mcomp[[idx]][2], 
      ", D:", D, ", df:", df, ", p-value:", p, "\n")
}
