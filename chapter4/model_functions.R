#----------------------------------------------------------#
# モデルの設定
#----------------------------------------------------------#

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

  # set Q values (#option x T)
  Q <- matrix(numeric(2*T), nrow=2, ncol=T)
  
  # initialize log-likelihood
  ll <- 0
  
  for (t in 1:T) {
    
    if (c[t] > 0) {
      # choosing prob A
      pA[t] <- 1/(1+exp(-beta * (Q[1,t]-Q[2,t])))
      
      ll <- ll + (c[t]==1) * log(pA[t]) +  (c[t]==2) * log(1-pA[t])
      
      # update values 
      if (t < T) {
        
        Q[c[t],t+1] <- Q[c[t],t] + alpha * (r[t] - Q[c[t],t] ) 
        
        # for unchosen option
        Q[3-c[t],t+1] <- Q[3-c[t],t]
      }
    } else {
      # update values 
      if (t < T) 
        Q[,t+1] <- Q[,t] 
    }
  }
  
  # log prior density 
  if (is.null(prior)) {
    lprior <- 0
  } else {
    lprior <- dbeta(alpha,prior$alpha_a, prior$alpha_b,log = T) + 
      dgamma(beta,shape=prior$beta_shape, scale=prior$beta_scale,log = T) 
  }
  
  return(list(negll = -ll - lprior,Q = Q, pA = pA))
}

# win-stay lose-shift
func_wsls <- function(param, data, prior = NULL)
{
  
  epsilon <- param[1] # error rate
  c <- data$choice
  r <- data$reward
  T <- length(c)
  
  pA <- numeric(T) 
  
  # initialize log-likelihood
  ll <- 0
  
  for (t in 1:T) {
   
    # choosing prob 1
    if (t == 1) {
      pA[t] <- 0.5
    } else {
      if (c[t-1] == 0) {
        pA[t] <- 0.5
      } else {
        if (r[t-1]==1)
          pA[t] <- (c[t-1]==1) * (1-epsilon) + (c[t-1]==2) * epsilon
        else 
          pA[t] <- (c[t-1]==1) * (epsilon) + (c[t-1]==2) * (1-epsilon)
      }
    }
    
    if (c[t] > 0)
      ll <- ll + (c[t]==1) * log(pA[t]) +  (c[t]==2) * log(1-pA[t])
  }
  
  # log prior density 
  if (is.null(prior)) {
    lprior <- 0
  } else {
    lprior <- dbeta(epsilon, prior$epsilon_a, prior$epsilon_b, log = T)
  }
  
  return(list(negll = -ll - lprior, pA = pA))
}

priorList <- list(
  list(alpha_a = 2, alpha_b = 2, beta_shape = 2, beta_scale = 3), # q-learning
  list(epsilon_a = 0.1, epsilon_b = 4),  # winstay-lose shift
  list(pA_a = 2, pA_b = 2) # biased random choice
)

# biased random choice
func_random <- function(param, data, prior = NULL)
{
  
  pA <- param[1] # choice prob
  c <- data$choice
  T <- length(c)
  
  # initialize log-likelihood
  ll <- 0
  
  ll <- sum(c == 1) * log(pA) + sum(c == 2) * log(1-pA) 
  
  
  # log prior density 
  if (is.null(prior)) {
    lprior <- 0
  } else {
    lprior <- dbeta(pA, prior$pA_a, prior$pA_b, log = T)
  }
  
  return(list(negll = -ll - lprior, pA = pA))
}


# 事前分布のハイパーパラメータの設定 
priorList <- list(
  list(alpha_a = 2, alpha_b = 2, beta_shape = 2, beta_scale = 3), # q-learning
  list(epsilon_a = 0.1, epsilon_b = 4),  # winstay-lose shift
  list(pA_a = 2, pA_b = 2)  # biased choice
)

# モデル関数のリスト 
modelfunctions <- c(func_qlearning, func_wsls, func_random)

# パラメータ数のリスト
nParamList <- c(2,1,1)

# モデル数
nModel <- length(nParamList)
