# メモリ，グラフのクリア
rm(list=ls())
graphics.off()

# 描画のためのライブラリ読み込み
library(plot3D)  

# 乱数のシードの設定 
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

# Q-learning
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

# 最適化により最小化する負の対数尤度を返す関数
func_minimize <- function(modelfunc, param, choice, reward)
{
  ret <- modelfunc(param, choice, reward)
  
  # 負の対数尤度のみ返す
  return(ret$negll)
}

#----------------------------------------------------------#
# 対数尤度関数の作成
#----------------------------------------------------------#

alpha <- seq(0.01,1.0, by = 0.01)
beta <- seq(0.1,4.0, by = 0.1)

llval <- matrix(NA, nrow = length(alpha), ncol = length(beta))

for (idxa in 1:length(alpha)) {
  for (idxb in 1:length(beta)) {
    llval[idxa,idxb] <- - func_minimize(c(alpha[idxa], beta[idxb]), modelfunc = func_qlearning, 
                                        choice=c, reward=r)
  }
}

llval <- pmax(llval,-100)

#----------------------------------------------------------#
# 対数尤度関数の3次元描画
#----------------------------------------------------------#

x11()
plot3D::persp3D(alpha, beta, llval, 
                phi = 25,
                theta = 40, 
                colvar = llval,
                col = "gray", 
                ticktype = "detailed", 
                xlab = "alpha",
                ylab = "beta",
                zlab = "Log-likelihood",
                zlim = c(-90,-40), 
                axes=TRUE, 
                contour = list(col = "black", nlevels = 25, side = c("zmin", "z"))
)
