# clear 
rm(list=ls())
graphics.off()
library(tidyverse)

set.seed(141)

#----------------------------------------------------------#
# Simulation for generating samples
#----------------------------------------------------------#

# number of simulation trials
T <- 80

# initialize Q values (#option x T)
Q <- matrix(numeric(2*T), nrow=2, ncol=T)

c <- numeric(T)
r <- numeric(T) 
p1 <- numeric(T) 

alpha <- 0.3     # learning rate
beta <- 2.0      # Inverse temperature

# reward probability for each option
pr <- c(0.7,0.3)

for (t in 1:T) {
  
  # choosing prob 1
  p1[t] <- 1/(1+exp(-beta*(Q[1,t]-Q[2,t])))
  
  if (runif(1,0,1) < p1[t]) {
    # choose option 1
    c[t] <- 1
    r[t] <- as.numeric(runif(1,0,1) < pr[1])
  } else {
    # choose option 2
    c[t] <- 2
    r[t] <- as.numeric(runif(1,0,1) < pr[2])
  }
  
  # update values 
  if (t < T) {
    
    Q[c[t],t+1] <- Q[c[t],t] + alpha * (r[t] - Q[c[t],t] ) 
    
    # for unchosen option
    Q[3-c[t],t+1] <- Q[3-c[t],t]
  }
}


#----------------------------------------------------------#
# Model function (for parameter fit)
#----------------------------------------------------------#

# Q-learning
func_qlearning <- function(param, choice, reward)
{
  T <- length(choice)
  alpha <- param[1]
  beta <- param[2]
  c <- choice
  r <- reward
  
  p1 <- numeric(T) 

  # set Q values (#option x T)
  Q <- matrix(numeric(2*T), nrow=2, ncol=T)
  
  # initialize log-likelihood
  ll <- 0
  
  for (t in 1:T) {
    
    # choosing prob 1
    p1[t] <- 1/(1+exp(-beta * (Q[1,t]-Q[2,t])))
    
    ll <- ll + (c[t]==1) * log(p1[t]) +  (c[t]==2) * log(1-p1[t])
    
    # update values 
    if (t < T) {
      
      Q[c[t],t+1] <- Q[c[t],t] + alpha * (r[t] - Q[c[t],t] ) 
      
      # for unchosen option
      Q[3-c[t],t+1] <- Q[3-c[t],t]
    }
  }
  return(list(negll = -ll,Q = Q, p1 = p1))
}


# win-stay lose-shift
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
    
    # choosing prob 1
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

#----------------------------------------------------------#
# objective function to be minimized (for parameter fit)
#----------------------------------------------------------#

func_minimize <- function(modelfunc, param, choice, reward)
{
  ret <- modelfunc(param, choice, reward)
  
  # return negative log-likelihood
  return(ret$negll)
}


#----------------------------------------------------------#
# Parameter fit 
#----------------------------------------------------------#

# q-learning
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

# parameter misspecification
# high alpha 
ret <- func_qlearning(c(0.9, beta), choice=c, reward=r)
Qh <- ret$Q
p1h <- ret$p1
# low alpha
ret <- func_qlearning(c(0.1, beta), choice=c, reward=r)
Ql <- ret$Q
p1l <- ret$p1


# winstay-lose shift
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

# biased random model 
p1est_random <- sum(c == 1)/length(c)
ll <- sum(c == 1) * log(p1est_random) + sum(c == 2) * log(1-p1est_random) 
print(sprintf("Model 3: log-likelihood: %.2f, AIC: %.2f", ll, -2*ll + 2))


source("plot_results_qlearning_simulation1.R")
# source("plot_results_qlearning_simulation2.R")

source("plot_ll_contour.R")

