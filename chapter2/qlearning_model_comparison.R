# clear 
rm(list=ls())
graphics.off()
library(tidyverse)
library(gridExtra)

set.seed(141)

#----------------------------------------------------------#
# Q学習モデルのシミュレーションによるデータ生成
#----------------------------------------------------------#

# 試行数
T <- 80

# Q 値の初期化( 選択肢の数x T)
Q <- matrix(numeric(2*T), nrow=2, ncol=T)

c <- numeric(T)
r <- numeric(T) 
p1 <- numeric(T) 

alpha <- 0.3     # 学習率
beta <- 2.0      # 逆温度

# それぞれの選択肢の報酬確率
pr <- c(0.7,0.3)

for (t in 1:T) {
  
  # ソフトマックスで選択肢A の選択確率を決定する
  p1[t] <- 1/(1+exp(-beta*(Q[1,t]-Q[2,t])))
  
  if (runif(1,0,1) < p1[t]) {
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


#----------------------------------------------------------#
# フィットするモデルの設定
#----------------------------------------------------------#

# Q学習モデル
func_qlearning <- function(param, choice, reward)
{
  T <- length(choice)
  alpha <- param[1]
  beta <- param[2]
  
  # 短い名前の変数に持ち替える
  c <- choice
  r <- reward
  
  p1 <- numeric(T) 
  
  # Q 値の初期化( 選択肢の数x T)
  Q <- matrix(numeric(2*T), nrow=2, ncol=T)
  
  # 対数尤度を格納する変数
  ll <- 0
  
  for (t in 1:T) {
    
    # ソフトマックスで選択肢A の選択確率を決定する
    p1[t] <- 1/(1+exp(-beta * (Q[1,t]-Q[2,t])))
    
    # 試行tの対数尤度は実際の選択がA (c=1) であれば log(p1[t]), 
    # B (c=2) であればlog(1 - p1[t]) となる
    ll <- ll + (c[t]==1) * log(p1[t]) +  (c[t]==2) * log(1-p1[t])
    
    # 行動価値の更新
    if (t < T) {
      
      Q[c[t],t+1] <- Q[c[t],t] + alpha * (r[t] - Q[c[t],t] ) 
      
      # 選択肢していない行動の価値はそのままの値を次の時刻に引き継ぐ
      Q[3-c[t],t+1] <- Q[3-c[t],t]
    }
  }
  return(list(negll = -ll,Q = Q, p1 = p1))
}

# WSLS (winstay-lose shift) モデル
func_wsls <- function(param, choice, reward)
{
  T <- length(choice)
  epsilon <- param[1] # error rate
  c <- choice
  r <- reward
  
  p1 <- numeric(T) 
  
  # initialize log-likelihood
  ll <- 0
  
  for (t in 1:T) {
    
    # 選択肢Aの選択確率を決定
    if (t == 1) {
      p1[t] <- 0.5
    } else {
      if (r[t-1]==1)
        p1[t] <- (c[t-1]==1) * (1-epsilon) + (c[t-1]==2) * epsilon
      else 
        p1[t] <- (c[t-1]==1) * (epsilon) + (c[t-1]==2) * (1-epsilon)
    }
    
    ll <- ll + (c[t]==1) * log(p1[t]) +  (c[t]==2) * log(1-p1[t])
  }
  return(list(negll = -ll, p1 = p1))
}

# 最適化により最小化する負の対数尤度を返す関数
func_minimize <- function(modelfunc, param, choice, reward)
{
  ret <- modelfunc(param, choice, reward)
  
  # return negative log-likelihood
  return(ret$negll)
}

#----------------------------------------------------------#
# 最適化による最尤推定の実行
#----------------------------------------------------------#

# Q学習モデル
fvalmin <- Inf
for (idx in 1:10) {
  
  # set initial value
  initparam <- runif(2, 0, 1.0)
  
  res <- optim(initparam, func_minimize,
              hessian = TRUE, modelfunc = func_qlearning, choice=c, reward=r)
  
  if (res$value < fvalmin) {
    paramest <- res$par
    fvalmin <- res$value
  }
}

print(sprintf("alpha - True value: %.2f, Estimated value: %.2f", alpha, paramest[1]))
print(sprintf("beta  - True value: %.2f, Estimated value: %.2f", beta, paramest[2]))
print(sprintf("Model 1: log-likelihood: %.2f, AIC: %.2f", -fvalmin, 2*fvalmin + 2*2))

ret <- func_qlearning(paramest, choice=c, reward=r)

Qest <- ret$Q
p1est <- ret$p1

# WSLS (winstay-lose shift) モデル
fvalmin <- Inf
for (idx in 1:10) {
  
  # set initial value
  initparam <- runif(1, 0, 1.0)
  
  res <- optim(initparam, func_minimize,
               hessian = TRUE, modelfunc = func_wsls, choice=c, reward=r)
  
  if (res$value < fvalmin) {
    paramest <- res$par
    fvalmin <- res$value
  }
}

ret <- func_wsls(paramest, choice=c, reward=r)
print(sprintf("Model 2: log-likelihood: %.2f, AIC: %.2f", -fvalmin, 2*fvalmin + 2))
p1est_wsls <- ret$p1

# ランダム選択モデル
p1est_random <- sum(c == 1)/length(c)
ll <- sum(c == 1) * log(p1est_random) + sum(c == 2) * log(1-p1est_random) 
print(sprintf("Model 3: log-likelihood: %.2f, AIC: %.2f", ll, -2*ll + 2))

#----------------------------------------------------------#
# 結果の描画
#----------------------------------------------------------#
ggplot() + theme_set(theme_bw(base_size = 18,base_family="Arial")) 

maxtrial <- 80

df <- data.frame(trials = 1:T, 
                 Q1est = Qest[1,],
                 Q2est = Qest[2,],
                 Q1 = Q[1,],
                 Q2 = Q[2,],
                 c = c,
                 r = as.factor(r),
                 p1 = p1,
                 p1est_wsls = p1est_wsls,
                 p1est = p1est, 
                 p1est_random = p1est_random)

dfplot <- df %>% filter(trials <= maxtrial)

# 選択確率のモデル間比較
x11(width = 7, height = 3)
g_p1 <- ggplot(df, aes(x = trials, y = p1est)) + 
  xlab("試行") + 
  ylab("P(a = A)") +
  geom_line(aes(y = p1est), linetype = 2, size=1.0) +
  geom_line(aes(y = p1est_wsls), linetype = 1, size=1, color="gray44") +
  geom_line(aes(y = p1est_random), linetype = 3, size=1, color="gray55") +
  geom_point(data = df %>% filter(c==1 & r == 1), 
             aes(x = trials, y = 1.12), shape = 25, size = 1.5) + 
  geom_point(data = df %>% filter(c==2 & r == 1), 
             aes(x = trials, y = -0.12), shape = 2, size = 1.5) + 
  theme(legend.position = "none") + 
  geom_linerange(data = df %>% filter(c==1), 
                 aes(
                   x = trials, 
                   ymin = 1.0, 
                   ymax = 1.05), 
                 size=1) + 
  scale_y_continuous(breaks=c(0, 0.5, 1.0), labels = c(0,0.5,1)) +
  geom_linerange(data = df %>% filter(c==2), 
                 aes(
                   x = trials, 
                   ymin = -0.05, 
                   ymax = 0.0), 
                 size=1) +
  geom_path(size = 1.2) 

print(g_p1)

# 図を保存する場合は以下を実行
# ggsave(file="./figs/qlarning_ll_comparison.eps", g_p1) 

