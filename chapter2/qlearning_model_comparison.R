# ----------------------------------------------------- #
# Q学習モデルのシミュレーションにより選択データを生成し，
# 対数尤度関数をプロットする。
# ----------------------------------------------------- #

# メモリ，グラフのクリア
rm(list=ls())
graphics.off()

# 描画のためのライブラリ読み込み
library(tidyverse)

set.seed(141)

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


# フィットするモデルの設定 ------------------------------------------------------------

# Q学習モデル
func_qlearning <- function(param, choice, reward)
{
  T <- length(choice)
  alpha <- param[1]
  beta <- param[2]
  
  # 短い名前の変数に持ち替える
  c <- choice
  r <- reward
  
  pA <- numeric(T) 
  
  # Q 値の初期化( 選択肢の数x T)
  Q <- matrix(numeric(2*T), nrow=2, ncol=T)
  
  # 対数尤度を格納する変数
  ll <- 0
  
  for (t in 1:T) {
    
    # ソフトマックスで選択肢A の選択確率を決定する
    pA[t] <- 1/(1+exp(-beta * (Q[1,t]-Q[2,t])))
    
    # pAが最適化の中で一時的にでも数値的に0や1になると対数尤度がNaNになり
    # 最適化に失敗することから，0.0001と0.9999の間に収まるようにする。
    # (最終的な結果にはほとんど影響しない)
    pA[t] <- min(max(pA[t], 0.0001), 0.9999)
    
    # 試行tの対数尤度は実際の選択がA (c=1) であれば log(pA[t]), 
    # B (c=2) であればlog(1 - pA[t]) となる
    ll <- ll + (c[t]==1) * log(pA[t]) +  (c[t]==2) * log(1-pA[t])
    
    # 行動価値の更新
    if (t < T) {
      
      Q[c[t],t+1] <- Q[c[t],t] + alpha * (r[t] - Q[c[t],t] ) 
      
      # 選択肢していない行動の価値はそのままの値を次の時刻に引き継ぐ
      Q[3-c[t],t+1] <- Q[3-c[t],t]
    }
  }
  return(list(negll = -ll,Q = Q, pA = pA))
}

# WSLS (winstay-lose shift) モデル
func_wsls <- function(param, choice, reward)
{
  T <- length(choice)
  epsilon <- param[1] # error rate
  c <- choice
  r <- reward
  
  pA <- numeric(T) 
  
  ll <- 0
  
  for (t in 1:T) {
    
    # 選択肢Aの選択確率を決定
    if (t == 1) {
      pA[t] <- 0.5
    } else {
      if (r[t-1]==1)
        pA[t] <- (c[t-1]==1) * (1-epsilon) + (c[t-1]==2) * epsilon
      else 
        pA[t] <- (c[t-1]==1) * (epsilon) + (c[t-1]==2) * (1-epsilon)
    }
    
    # pAが最適化の中で一時的にでも数値的に0や1になると対数尤度がNaNになり
    # 最適化に失敗することから，0.0001と0.9999の間に収まるようにする。
    # (最終的な結果にはほとんど影響しない)
    pA[t] <- min(max(pA[t], 0.0001), 0.9999)
    
    ll <- ll + (c[t]==1) * log(pA[t]) +  (c[t]==2) * log(1-pA[t])
  }
  return(list(negll = -ll, pA = pA))
}

# 最適化により最小化する負の対数尤度を返す関数
func_minimize <- function(modelfunc, param, choice, reward)
{
  ret <- modelfunc(param, choice, reward)
  
  return(ret$negll)
}

# 最尤推定 --------------------------------------------------------------------

# Q学習モデル
fvalmin <- Inf
for (idx in 1:10) {
  
  # set initial value
  initparam <- runif(2, 0, 1.0)
  
  res <- optim(initparam, func_minimize,
              modelfunc = func_qlearning, 
              choice=c, reward=r)
  
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
pAest <- ret$pA

# WSLS (winstay-lose shift) モデル

# 最適化するパラメータが1つの場合は
# optimではなくoptimizeを使うことが推奨されている
res <- optimize(func_minimize, c(0,1), 
                modelfunc = func_wsls, 
                choice=c, reward=r)

paramest <- res$minimum
fvalmin <- res$objective

ret <- func_wsls(paramest, choice=c, reward=r)
print(sprintf("Model 2: log-likelihood: %.2f, AIC: %.2f", -fvalmin, 2*fvalmin + 2))
pAest_wsls <- ret$pA

# ランダム選択モデル
pAest_random <- sum(c == 1)/length(c)
ll <- sum(c == 1) * log(pAest_random) + sum(c == 2) * log(1-pAest_random) 
print(sprintf("Model 3: log-likelihood: %.2f, AIC: %.2f", ll, -2*ll + 2))


# 結果の描画 -------------------------------------------------------------------

ggplot() + theme_set(theme_bw(base_size = 18)) 

maxtrial <- 80

df <- data.frame(trials = 1:T, 
                 Q1est = Qest[1,],
                 Q2est = Qest[2,],
                 Q1 = Q[1,],
                 Q2 = Q[2,],
                 c = c,
                 r = as.factor(r),
                 pA = pA,
                 pAest_wsls = pAest_wsls,
                 pAest = pAest, 
                 pAest_random = pAest_random)

dfplot <- df %>% filter(trials <= maxtrial)

# 選択確率のモデル間比較
x11(width = 7, height = 3)
g_pA <- ggplot(df, aes(x = trials, y = pAest)) + 
  xlab("試行") + 
  ylab("P(a = A)") +
  geom_line(aes(y = pAest), linetype = 2, size=1.0) +
  geom_line(aes(y = pAest_wsls), linetype = 1, size=1, color="gray44") +
  geom_line(aes(y = pAest_random), linetype = 3, size=1, color="gray55") +
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

print(g_pA)

# 図を保存する場合は以下を実行
# ggsave(file="./figs/qlarning_ll_comparison.eps", g_pA) 

