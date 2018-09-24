#----------------------------------------------------------#
# 例6-2のデータ生成
#----------------------------------------------------------#

# メモリのクリア
rm(list=ls())

# ライブラリの読み込み
library(tidyverse)


# シミュレーションの設定部 ------------------------------------------------------------

# シミュレーションのID
simulation_ID <- "Qlearning_trait_correlation_random_slope" 

# データを保存するファイル名
csv_simulation_data <- paste0("./data/simulation_data", simulation_ID, ".csv")
csv_param <- paste0("./data/trueparam_",simulation_ID, ".csv")

# 参加者数
nSub <- 40

# 一人の参加者あたりの試行数 
T <- 100

# グループ数
nGroup <- 1 

beta.mean <- 2.0
beta.sigma <- 0.5

# 乱数のシードを設定
set.seed(12)

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
  cat("Generating data for group ", idxGroup, "...\n")
  
  # 回帰モデルによる学習率の決定
  trait <- rnorm(nSub,0,1)
  slope <- rnorm(nSub,1,1)

  h <- 0.2 + slope*trait + rnorm(nSub,0,1.0)
  alphaL <- 1/(1+exp(-h))

  beta <- rnorm(n = nSub, mean = beta.mean[idxGroup], sd = beta.sigma[idxGroup])
  
  # 範囲内におさめる
  alphaL <- pmax(pmin(alphaL,1.0),0.01) 
  beta <- pmax(pmin(beta,10.0),0.01) 
  
  alphaF <- numeric(nSub) 
  
  # 学習率と特性の散布図
  df <- data.frame(trait=trait, alpha=alphaL)
  ggplot()+theme_set(theme_bw(base_size = 14))
  # x11()
  g <- ggplot(df, aes(x=trait,y=alpha)) + 
    geom_point(shape=0, size = 3) +
    scale_shape_manual(values=c(0, 17)) +
    theme(axis.text.x = element_text(size=20),axis.text.y = element_text(size=20)) +
    xlab("症状スコア") +
    ylab("学習率") + 
    coord_cartesian(ylim =c(0,1))
  print(g)
  # ggsave(file= paste0("./figs/trait_parameter_correlation.eps"), g)
  
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
                        # choice_trace = choice_trace, 
                        Q1 = Q[1,], Q2 = Q[2,], 
                        PA = pA)
    
    df_simulation_data <- dplyr::bind_rows(df_simulation_data, dftmp)
    
    # record true parameter values into dataframe
    dftmp_param <- data.frame(group = as.factor(idxGroup), 
                        subject = as.factor(idxSub), 
                        trait = trait[idxSub], 
                        alphaL = alphaL[idxSub],
                        alphaF = alphaF[idxSub], 
                        beta = beta[idxSub])
    
    dfparam <- rbind(dfparam, dftmp_param)
    
  }
}


# 生成したデータをcsvファイルとして保存する --------------------------------------------------

# シミュレーションにより生成した選択データ
write.table(df_simulation_data , file = csv_simulation_data,
            quote = FALSE, sep = ",",row.names = FALSE)

# パラメータの真値
write.table(dfparam, file = csv_param, 
            quote = FALSE, sep = ",",row.names = FALSE)

