#----------------------------------------------------------#
# 例6-1のデータ生成
#----------------------------------------------------------#

# メモリのクリア
rm(list=ls())

# ライブラリの読み込み
library(tidyverse)

# シミュレーションの設定部 ------------------------------------------------------------

# シミュレーションのID
simulation_ID <- "Qlearning_group_comparison" 

# データを保存するファイル名
csv_simulation_data <- paste0("./data/simulation_data", simulation_ID, ".csv")
csv_param <- paste0("./data/trueparam_",simulation_ID, ".csv")

# 各グループの参加数
nSub <- 20

# 一人の参加者あたりの試行数 
T <- 100

# 乱数のシードを設定
set.seed(4)

# 集団レベル分布の平均 (グループが複数ある場合はベクトルとして並べる。例: c(0.2, 0.4))
alphaL.mean <- c(0.3, 0.5)  # 学習率
beta.mean <- c(2.0, 2.0)    # 逆温度

# グループ数を求める (nGroup = 2)
nGroup <- length(alphaL.mean)

# 集団レベル分布の標準偏差
alpha.sigma <- rep(0.1, nGroup) # 学習率
beta.sigma <- rep(0.5, nGroup)  #  逆温度

# 報酬確率の系列を作る
pr <- list()
interval <- 50  # 同じ報酬確率が続く試行数
padv <- 0.7     # 良い選択肢の報酬確率
pr[[1]] <- rep(rep(c(padv,1-padv),T/interval/2), each=interval)
pr[[2]] <- rep(rep(c(1-padv,padv),T/interval/2), each=interval)


# シミュレーションの実行 -------------------------------------------------------------

# パラメータの真値を格納するデータフレームの準備
dfparam <- data.frame()

# 選択などのデータを格納するデータフレームの準備
df_simulation_data <- data.frame()

for (idxGroup in 1:nGroup) {
  # グループごとに...
  cat("Generating data for group ", idxGroup, "...\n")
  
  # 学習率の真値の設定
  alphaL <- rnorm(n = nSub, mean = alphaL.mean[idxGroup], sd = alpha.sigma[idxGroup])
  alphaL <- pmax(pmin(alphaL,1.0),0.01) # 0.01以上1.0以下におさめる
  alphaF <- numeric(nSub)               # 標準的Q学習モデルではalphaF = 0に
  
  alphaF <- alphaL                      # F-Q学習モデルではalphaL = alphaF
  
  # 逆温度の真値の設定
  beta <- rnorm(n = nSub, mean = beta.mean[idxGroup], sd = beta.sigma[idxGroup])
  beta <- pmax(pmin(beta,10.0),0.01)    # 0.01以上10.0以下におさめる
  
  
  # パラメータの真値を記録
  dftmp <- data.frame(group = as.factor(idxGroup), 
                      subject = as.factor(1:nSub), 
                      alphaL = alphaL,
                      alphaF = alphaF, 
                      beta = beta)
  
  # データフレームを縦に連結していく
  dfparam <- rbind(dfparam, dftmp)
  
  for (idxSub in 1:nSub) {
    
    # 変数の初期化
    Q <- matrix(numeric(2*T), nrow=2, ncol=T)
    c <- numeric(T)
    r <- numeric(T) 
    pA <- numeric(T) 
    
    for (t in 1:T) {
      
      # ソフトマックスで選択肢A の選択確率を決定する
      pA[t] <- 1/( 1 + exp(- (beta[idxSub] * (Q[1,t]-Q[2,t]))) )
      
      if (runif(1,0,1) < pA[t]) {
        # Aを選択
        c[t] <- 1
        r[t] <- as.numeric(runif(1,0,1) < pr[[1]][t])
      } else {
        # Bを選択
        c[t] <- 2
        r[t] <- as.numeric(runif(1,0,1) < pr[[2]][t])
      }
      
      # 行動価値の更新
      if (t < T) {
        delta <- r[t] - Q[c[t],t]
        
        Q[c[t],t+1] <- Q[c[t],t] + alphaL[idxSub] * delta
        
        Q[3-c[t],t+1] <- (1-alphaF[idxSub]) * Q[3-c[t],t]
      }
    }
    
    # 一時保管用のデータフレームに格納する
    dftmp <- data.frame(group = as.factor(idxGroup), 
                        subject = idxSub, 
                        trial = 1:T,
                        choice = c, 
                        reward = r,
                        Q1 = Q[1,], Q2 = Q[2,], 
                        pA = pA)
    
    # データフレームを縦に連結していく
    df_simulation_data <- rbind(df_simulation_data, dftmp)
  }
}


# 生成したデータをcsvファイルとして保存する --------------------------------------------------

# シミュレーションにより生成した選択データ
write.table(df_simulation_data , file = csv_simulation_data,
            quote = FALSE, sep = ",",row.names = FALSE)

# パラメータの真値
write.table(dfparam, file = csv_param, 
            quote = FALSE, sep = ",",row.names = FALSE)


# パラメータ真値のプロット ------------------------------------------------------------

ggplot()+theme_set(theme_bw(base_size = 14))
# x11()
g <- ggplot(dfparam, aes(x=alphaL,y=beta, group=group)) + 
  geom_point(aes(shape=group), size = 3) +
  scale_shape_manual(values=c(0, 17)) +
  theme(axis.text.x = element_text(size=20),axis.text.y = element_text(size=20)) +
  xlab("学習率") +
  ylab("逆温度") + 
  coord_cartesian(xlim = c(0,1), ylim =c(0,3))
print(g)

# filename <- paste0("./figs/trueparam_",simulation_ID, ".eps")
# ggsave(file=filename, g) #saves g

