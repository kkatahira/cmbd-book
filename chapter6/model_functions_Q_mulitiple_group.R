#----------------------------------------------------------#
# Model functions (for parameter fit)
#----------------------------------------------------------#

# standard Q-learning
func_qlearning <- function(param, gpmap, idxGroup = 1, data, prior = NULL)
{
  alpha <- param[gpmap[idxGroup,1]]
  beta <- param[gpmap[idxGroup,2]]
  c <- data$choice
  r <- data$reward
  T <- length(c)
  
  pA <- numeric(T) 

  # set Q values (#option x T)
  Q <- matrix(numeric(2*T), nrow=2, ncol=T)
  
  # initialize log-likelihood
  ll <- 0
  
  for (t in 1:T) {
    
    pA[t] <- 1/(1+exp(-beta * (Q[1,t]-Q[2,t])))
    
    pA[t] <- max(min(pA[t], 0.9999), 0.0001)
    
    ll <- ll + (c[t]==1) * log(pA[t]) +  (c[t]==2) * log(1-pA[t])
    
    # update values 
    if (t < T) {
      
      Q[c[t],t+1] <- Q[c[t],t] + alpha * (r[t] - Q[c[t],t] ) 
      
      # for unchosen option
      Q[3-c[t],t+1] <- Q[3-c[t],t]
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


modelfunctions <- c(func_qlearning,
                    func_qlearning,
                    func_qlearning,
                    func_qlearning) 

# gpmap[idxGroup,idxParam] が，全体のパラメータベクトル paramの何番目かを指定する
# gpmap[1,1] = gpmap[2,1] = 2のとき，1番目のパラメータalphaは，2グループで共通
# gpmap[1,1] = 1, gpmap[2,1] = 2のとき，1番目のパラメータalphaは，2グループで異なる

gpmap <- list(matrix(c(1,1,2,2),nrow = 2, ncol = 2), 
             matrix(c(1,2,3,3),nrow = 2, ncol = 2), 
             matrix(c(1,1,2,3),nrow = 2, ncol = 2),
             matrix(c(1,2,3,4),nrow = 2, ncol = 2))

priorList <- list(
  list(alpha_a = 2, alpha_b = 2, beta_shape = 2, beta_scale = 3) # q-learning
)

nParamList <- c(2,3,3,4)

nModel <- length(gpmap)
