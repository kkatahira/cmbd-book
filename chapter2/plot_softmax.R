# メモリとグラフのクリア
rm(list=ls())
graphics.off()

library(ggplot2)
ggplot() + theme_set(theme_bw(base_size = 25)) 

x11()

# 逆温度betaの値
beta_plot <- c(2.0, 1, 5) 

# ソフトマックス関数の定義
smax <- function(x, beta) {
  return(1/(1 + exp( - beta * x)))
}

x11()
g <- ggplot(data.frame(X = c(-4, 4)), aes(x=X)) +
  mapply(
    function(beta, lt) stat_function(fun = smax, args = beta, linetype = lt, size = 1.5),
    beta_plot, c(1,2,3)
  ) + 
  ylab('行動Aの選択確率 P(a = A)') + 
  xlab('Q(A) - Q(B)')
print(g)

# 保存するときは以下をコメントアウト
# ggsave(file = paste0("../figs/softmax.eps"), g) 
