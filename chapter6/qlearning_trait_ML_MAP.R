#----------------------------------------------------------#
# パラメータの点推定値 (最尤推定，MAP推定) を用いた
# 特性との相関分析
#----------------------------------------------------------#

# メモリのクリア
rm(list=ls())
graphics.off()

# ライブラリの読み込み
library(tidyverse)

# 乱数のシードを設定
set.seed(141)

# モデル関数の読み込み
source("model_functions_Q_mulitiple_group.R")

# パラメータ推定用関数の読み込み
source("parameter_fit_functions_multiple_group.R")

# 読み込むデータのsimulation ID
simulation_ID <- "Qlearning_trait_correlation_random_slope" # simulation ID for file name

# 読み込むデータファイルの指定
csv_simulation_data <- paste0("./data/simulation_data", simulation_ID, ".csv")

# データファイルの読み込み---------------
data <- read.table(csv_simulation_data, header = T, sep = ",")

# パラメータファイル名の指定，読み込み
csv_param <- paste0("./data/trueparam_",simulation_ID, ".csv")
data_param <- read.table(csv_param, header = T, sep = ",")


# モデルフィッティング，相関の検定 --------------------------------------------------------

cat("----------- single subject  ML -----------\n")
# 個人レベル最尤推定 
nParamList <- c(2)
resultSSML <- paramfitSSML(modelfunctions[1], data, nParamList)

# 学習率の相関の検定
print(cor.test(resultSSML$param.1, data_param$trait))

df <- data.frame(trait=data_param$trait, alpha=resultSSML$param.1)
ggplot()+theme_set(theme_bw(base_size = 14))
x11()
g <- ggplot(df, aes(x=trait,y=alpha)) + 
  geom_point(shape=0, size = 3) +
  scale_shape_manual(values=c(0, 17)) +
  theme(axis.text.x = element_text(size=20),axis.text.y = element_text(size=20)) +
  xlab("trait") +
  ylab("alpha") + 
  coord_cartesian(ylim =c(0,1))
print(g)


# 個人レベルMAP推定
cat("----------- MAP -----------\n")
nParamList <- c(2)
resultSSMAP <- paramfitSSMAP(modelfunctions[1], data, nParamList, priorList)

# 学習率の相関の検定
print(cor.test(resultSSMAP$param.1, data_param$trait))

df <- data.frame(trait=data_param$trait, alpha=resultSSMAP$param.1)
ggplot()+theme_set(theme_bw(base_size = 14))
x11()
g <- ggplot(df, aes(x=trait,y=alpha)) + 
  geom_point(shape=0, size = 3) +
  scale_shape_manual(values=c(0, 17)) +
  theme(axis.text.x = element_text(size=20),axis.text.y = element_text(size=20)) +
  xlab("trait") +
  ylab("alpha") + 
  coord_cartesian(ylim =c(0,1))
print(g)
