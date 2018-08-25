# メモリとグラフのクリア
rm(list=ls())
graphics.off()

# 描画のためのライブラリ読み込み
library(ggplot2)

#----------------------------------------------------------#
# RW modelのシミュレーションの実行
#----------------------------------------------------------#

# 乱数のシードの設定 
set.seed(928)

# 試行数
T <- 50

# 学習率
alpha <- 0.2

# 報酬確率
pr <- 0.8

# 連合強度( 価値) V と報酬r の配列を作成
V <- numeric(T)
r <- numeric(T)

for (t in 1:T) {
  
  # 報酬の有無の決定
  r[t] <- as.numeric( runif(1 ,0 ,1) < pr)
  
  # 連合強度の更新
  if (t < T) {
    V[t +1] <- V[t] + alpha * (r[t] - V[t])
  }
}

#----------------------------------------------------------#
# グラフの描画
#----------------------------------------------------------#

# ggplotのテーマ設定
ggplot() + theme_set(theme_bw(base_size = 18)) 

# ウィンドウの生成
x11()

# 描画するデータをデータフレームにまとめる
dfplot <- data.frame(trials = 1:T, 
                 V = V,
                 r = as.factor(r))

# 描画
g <- ggplot(dfplot, aes(x = trials, y = V)) +
  geom_line(aes(y = V), linetype = 1, size=1.2) +
  geom_point(data = dfplot %>% filter(r == 1), 
             aes(x = trials, y = 1.12), shape = 25, size = 1.5) + 
  scale_y_continuous(breaks=c(0, 0.5, 1.0), labels = c(0,0.5,1)) +
  xlab("試行") +
  ylab("連合強度 V") 

print(g)
