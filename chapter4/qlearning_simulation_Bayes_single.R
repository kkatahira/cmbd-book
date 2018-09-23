# ----------------------------------------------------- #
# Q学習モデルのシミュレーションにより一人分の
# 選択データを生成し，そのデータからベイズ推定，MAP推定をする。
# (比較のため最尤推定も行う)
# ----------------------------------------------------- #

# メモリ，グラフのクリア
rm(list=ls())
graphics.off()

# ライブラリ読み込み
library(tidyverse)
library(gridExtra)
library(rstan)  

# 乱数のシードの設定
set.seed(141)

# モデル関数の定義 ----------------------------------------------------------------

# Q-learning
func_qlearning <- function(param, data, prior = NULL)
{
  
  alpha <- param[1]
  beta <- param[2]
  c <- data$choice
  r <- data$reward
  T <- length(c)
  
  pA <- numeric(T) 
  
  Q <- matrix(numeric(2*T), nrow=2, ncol=T)
  
  ll <- 0
  
  for (t in 1:T) {
    
    if (c[t] > 0) {
      
      pA[t] <- 1/(1+exp(-beta * (Q[1,t] - Q[2,t])))
      
      ll <- ll + (c[t]==1) * log(pA[t]) +  (c[t]==2) * log(1-pA[t])
      
      if (t < T) {
        
        Q[c[t],t+1] <- Q[c[t],t] + alpha * (r[t] - Q[c[t],t] ) 
        
        Q[3-c[t],t+1] <- Q[3-c[t],t]
      }
    } else {
      if (t < T) 
        Q[,t+1] <- Q[,t] 
    }
  }
  
  # 対数事後確率密度
  if (is.null(prior)) {
    lprior <- 0
  } else {
    lprior <- dbeta(alpha,prior$alpha_a, prior$alpha_b,log = T) + 
      dgamma(beta,shape=prior$beta_shape, scale=prior$beta_scale,log = T) 
  }
  
  return(list(negll = -ll - lprior,Q = Q, pA = pA))
}

# 事前分布のリスト
priorList <- 
  list(alpha_a = 2, alpha_b = 2, beta_shape = 2, beta_scale = 3)

# パラメータ数
nParam <- 2

# パラメータ推定のための関数 -----------------------------------------------------------

#　負の対数尤度のみを返すラッパー関数
func_minimize <- function(modelfunc, param, data, prior)
{
  ret <- modelfunc(param, data, prior)
  
  # 負の対数尤度のみを返す
  return(ret$negll)
}

# 最尤推定 
paramfitML <- function(modelfunction, data, nParam)
{
  
  fvalmin <- Inf
  
  for (idx in 1:10) {
    
    initparam <- runif(nParam, 0, 1.0)
    
    res <- optim(initparam, func_minimize,
                 hessian = TRUE, modelfunc = modelfunction, 
                 data=data, prior = NULL)
    
    if (res$value < fvalmin) {
      paramest <- res$par
      fvalmin <- res$value
    }
  }
  aic <- 2*fvalmin + 2 * nParam
  bic <- 2*fvalmin + nParam * log(T)
  negll <- fvalmin
  paramlist <- paramest
  
  print(sprintf("Estimated value: %.2f", paramest))
  print(sprintf("log-likelihood: %.2f, AIC: %.2f, BIC: %.2f", 
                negll, 
                aic,
                bic))
  return(list(negll = negll, aic = aic, bic = bic, paramest = paramest))
}


# MAP推定 
paramfitMAP <- function(modelfunction, data,  nParam, prior)
{
  
  fvalmin <- Inf
  
  for (idx in 1:10) {
    
    initparam <- runif(nParam, 0, 1.0)
    
    res <- optim(initparam, func_minimize,
                 hessian = TRUE, modelfunc = modelfunction, 
                 data=data, prior = prior)
    
    if (res$value < fvalmin) {
      paramest <- res$par
      fvalmin <- res$value
      lp <- -fvalmin
      H <- res$hessian
    }
  }
  negll <- fvalmin
  paramlist <- paramest
  
  # ラプラス近似で対数周辺尤度を求める
  lml <- lp + nParam/2 * log(2*pi) - 0.5 * log(det(H)) 
  
  print(sprintf("Estimated value: %.2f", paramest))
  print(sprintf("log marginal likelihood: %.2f", lml))
  
  return(list(negll = negll, lml = lml, paramest = paramest))
}


# Q学習モデルのシミュレーションによるデータ生成 -------------------------------------------------

# 試行数
T <- 80

# Q 値の初期化( 選択肢の数x T)
Q <- matrix(numeric(2*T), nrow=2, ncol=T)

c <- numeric(T)
r <- numeric(T) 
pA <- numeric(T) 

alpha <- 0.3     # 学習率
beta <- 2.0      # 逆温度

# それぞれの選択肢の報酬確率
pr <- c(0.7,0.3)

for (t in 1:T) {
  
  # ソフトマックスで選択肢A の選択確率を決定する
  pA[t] <- 1/(1+exp(-beta*(Q[1,t]-Q[2,t])))
  
  if (runif(1,0,1) < pA[t]) {
    # Aを選択
    c[t] <- 1
    r[t] <- as.numeric(runif(1,0,1) < pr[1])
  } else {
    # Bを選択
    c[t] <- 2
    r[t] <- as.numeric(runif(1,0,1) < pr[2])
  }
  
  # 行動価値の更新
  if (t < T) {
    
    Q[c[t],t+1] <- Q[c[t],t] + alpha * (r[t] - Q[c[t],t] ) 
    
    # 選択肢していない行動の価値はそのままの値を次の時刻に引き継ぐ。
    # 3-c でc=1 なら2, c=2 なら1, というように
    # 逆側の選択肢のインデックスが求まる。
    Q[3-c[t],t+1] <- Q[3-c[t],t]
  }
}

data <- list(reward=r,choice=c)

# 最尤推定 --------------------------------------------------------------------
cat("----------- ML -----------\n")
resultML <- paramfitML(func_qlearning, data, nParam)


# MAP推定 -------------------------------------------------------------------
cat("----------- MAP -----------\n")
resultMAP <- paramfitMAP(func_qlearning, data, nParam, priorList)


# ベイズ推定 (MCMC) ------------------------------------------------------------

# Stan用にデータをリストにまとめる
dataList = list(    
  c = c,
  r = r,
  T = T
)

# MCMC
stanFit <- stan(file = 'qlearning_single_subject.stan', 
                data = dataList, 
                iter = 5000, 
                thin = 1, 
                chains = 3)

# 以下でRhatなどを表示し，サンプリングの精度を評価。
# print(stanFit)


# 結果の描画 -------------------------------------------------------------------

x11(width = 14*2, height = 12*2)

# 事前分布 (alpha)
galpha_prior <- ggplot(data.frame(alpha = c(0, 1)), 
                       aes(x = alpha)) + 
  stat_function(fun = dbeta, args=list(2, 2),linetype = 1, size = 1.5) +
  ylab('density') +
  xlab('alpha') 

# 事前分布 (beta)
gbeta_prior <- ggplot(data.frame(beta = c(0, 20)), 
                      aes(x = beta)) + 
  stat_function(fun = dgamma, args=list(shape=2, scale=3),linetype = 1, size = 1.5) +
  ylab('density') +
  xlab('beta') 

# MCMCサンプルの抽出
df_post <- data.frame(rstan::extract(stanFit,"alpha"),
                       rstan::extract(stanFit,"beta"))

# 事後分布のプロット (alpha)
galpha_posterior <- ggplot(df_post,
            aes(x = alpha)) + 
  geom_histogram(aes(y = ..density..)) + 
  geom_density(size=1,linetype=1) +
  geom_vline(xintercept = resultMAP$paramlist[[1]][1], 
             size=1,linetype=2) + # MAP estimates
  geom_vline(xintercept = resultML$paramlist[[1]][1], 
             size=0.5,linetype=3) + # ML estimates
  ylab('density') +
  xlab('alpha') 

# 事後分布のプロット (beta)
gbeta_posterior <- ggplot(df_post,
            aes(x = beta)) + 
  geom_histogram(aes(y = ..density..)) + 
  geom_density(size=1,linetype=1) + 
  geom_vline(xintercept = resultMAP$paramlist[[1]][2], 
             size=1,linetype=2) + # MAP estimates
  geom_vline(xintercept = resultML$paramlist[[1]][2], 
             size=0.5,linetype=3) + # ML estimates
  ylab('density') +
  xlab('beta') 

grid.arrange(galpha_prior, gbeta_prior, 
             galpha_posterior, gbeta_posterior, ncol = 2)

# 図を保存する場合は以下を実行
# g <- arrangeGrob(galpha_prior, gbeta_prior, 
#                 galpha_posterior, gbeta_posterior, ncol = 2)
# ggsave(file="./figs/posterior_single_qlearning.eps", g) 

