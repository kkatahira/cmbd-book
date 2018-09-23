# ----------------------------------------------------- #
# ベルヌーイ試行 (コイントスの例) のベイズ推定
# ----------------------------------------------------- #

# メモリ，グラフのクリア
rm(list=ls())
graphics.off()

# 描画のためのライブラリ読み込み
library(tidyverse)
library(gridExtra)

# データの設定 ------------------------------------------------------------------

# 3回オモテが出たとする
data <- c(1,1,1)

# ベイズ推定に用いる関数の設定 --------------------------------------------------

# 事前分布 (ベータ分布) のハイパーパラメータの設定
a <- 5
b <- 5

# 尤度関数の定義 (muはオモテが出る確率)
likelihood <- function(mu, data) {
  C1 <- sum(data)   # 1を数える
  C0 <- sum(1-data) # 0を数える
  return(exp(C1 * log(mu) + C0 * log(1-mu)))
}

# 事前分布 (ベータ分布)
prior <- function(mu, a, b) {
  return( dbeta(mu, a, b, log = F) )
}

# 事後確率密度
# posterior (unnormalized)
post <- function(mu, data, a, b) {
  
  C1 <- sum(data)   # 1を数える
  C0 <- sum(1-data) # 0を数える
  
  return( dbeta(mu, a+C1, b+C0, log = F) )
}

# 結果の描画 -------------------------------------------------------------------

x11()

# ggplot2のテーマ設定
ggplot() + theme_set(theme_bw(base_size = 18)) 

# 尤度関数のプロット
gl <- ggplot(data.frame(mu = c(0, 1)), aes(x=mu)) +
  stat_function(fun = likelihood, 
                args = list(data=data), linetype = 1, size = 1.5) +
  ylab('pdf') +
  xlab('mu') +
  ggtitle("likelihood") +
  scale_y_continuous(breaks=c(0, 1), labels = c(0, 1)) 

# 事前分布のプロット
gprior <- ggplot(data.frame(mu = c(0, 1)), aes(x=mu)) +
  stat_function(fun = prior, 
                args = list(a=a, b=b), linetype = 1, size = 1.5) +
  ylab('pdf') +
  xlab('mu') +
  ggtitle("prior")  + 
  scale_y_continuous(breaks=c(0, 1, 2, 3), labels = c(0, 1, 2, 3)) 

# 事後分布のプロット
gp <- ggplot(data.frame(mu = c(0, 1)), aes(x=mu)) +
  stat_function(fun = post, 
                args = list(data=data, a=a, b=b), linetype = 1, size = 1.5) +
  ylab('pdf') +
  xlab('mu') +
  ggtitle("poterior") + 
  scale_y_continuous(breaks=c(0, 1, 2, 3), labels = c(0, 1, 2, 3))

# 3つの図を並べる
grid.arrange(gl, gprior, gp, nrow=3) 

# 図を保存する場合は以下を実行
# g <- arrangeGrob(gl, gp,gp, nrow=2) #generates g
# ggsave(file="./figs/bernoulli_bayes.eps", g) #saves g

