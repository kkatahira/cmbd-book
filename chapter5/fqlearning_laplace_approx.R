# clear 
rm(list=ls())
graphics.off()

source("model_functions_FQ.R")
source("parameter_fit_functions_rsolnp.R")

simulation_ID <- "FQlearning_siglesubject" # simulation ID for file name

csv_simulation_data <- paste0("./simulation_data/simulation_data", simulation_ID, ".csv")

dt <- read.table(csv_simulation_data, header = T, sep = ",")
data <- list(reward = dt$r, choice = dt$choice) 

# plot likelihood (F-Q) ================================= # 

# alphaL <- seq(0.01,0.99, by = 0.01)
alphaL <- seq(0.0,1, by = 0.01)

llval <- numeric(length = length(alphaL))

for (idxa in 1:length(alphaL)) {
  llval[idxa] <- - func_minimize(c(alphaL[idxa], 2.0), 
                                      modelfunc = func_fqlearning, 
                                      data = data, 
                                      prior = NULL #priorList[[3]]
  )
}

x11()
plot(alphaL,exp(llval),
     type = "l", 
     lwd = 2,
     xlab = "alpha", ylab = "likelihood",
     main = sprintf("F-Q learning: log.ml = %.2f / max.ll = %.2f", 
                    mean(log(exp(llval))),
                    max(llval)),
     cex      = 1.2,
     cex.lab  = 1.2,
     cex.axis = 1.2,
     cex.main = 1.2)

#----------------------------------------------------------#
# Parameter fit MAP 
#----------------------------------------------------------#

# forgetting Q-learning (fixed beta)
func_fqlearning <- function(param, data, prior = NULL)
{
  
  alpha <- param[1]
  alphaF <- param[1]
  beta <- 2.0
  c <- data$choice
  r <- data$reward
  T <- length(c)
  
  p1 <- numeric(T) 
  
  # set Q values (#option x T)
  Q <- matrix(numeric(2*T), nrow=2, ncol=T)
  
  # initialize log-likelihood
  ll <- 0
  
  for (t in 1:T) {
    
    # choosing prob 1
    p1[t] <- 1/(1+exp(-beta * (Q[1,t]-Q[2,t])))
    p1[t] <- max(min(p1[t], 0.9999), 0.0001)
    
    ll <- ll + (c[t]==1) * log(p1[t]) +  (c[t]==2) * log(1-p1[t])
    
    # update values 
    if (t < T) {
      
      Q[c[t],t+1] <- Q[c[t],t] + alpha * (r[t] - Q[c[t],t] ) 
      
      # for unchosen option
      Q[3-c[t],t+1] <- (1 - alphaF) * Q[3-c[t],t]
    }
  }
  
  # log prior density 
  if (is.null(prior)) {
    lprior <- 0
  } else {
    lprior <- dbeta(alpha,prior$alpha_a, prior$alpha_b,log = T) 
  }
  
  return(list(negll = -ll - lprior,Q = Q, p1 = p1))
}

modelfunctions <- c(func_fqlearning)

nParamList <- c(1)

priorList <- list(
  list(alpha_a = 1, alpha_b = 1)
)

nModel <- length(nParamList)

#-------solver----------------------------
library(Rsolnp)
nModel <- length(modelfunctions)
lml <- numeric(nModel)
negll <- numeric(nModel)
paramlist <- list()
hessian <- list()
prior <- priorList

idxm <- 1
fvalmin <- Inf
print(sprintf("Model %d:", idxm))

for (idx in 1:10) {
  
  # set initial value
  initparam <- runif(nParamList[idxm], 0, 1.0)
  
  res <- solnp(initparam, fun = func_minimize, 
               # LB = lblist[[idxmodel]], UB = ublist[[idxmodel]], 
               modelfunc = modelfunctions[[idxm]], 
               control = list(trace = 0),
               data = data, prior = prior[[idxm]])
  
  nll <- res$values[length(res$values)]
  
  if (nll < fvalmin) {
    paramest <- res$par
    # fvalmin <- nll
    lp <- -nll
    H <- res$hessian
  }
}
negll[idxm] <- fvalmin
paramlist[[idxm]] <- paramest

# log marginal likelihood (Laplace)
#lml[idxm] <- lp + nParam[idxm]/2 * log(nParam[idxm]*pi) - 0.5 * log(det(H)) 
lml[idxm] <- lp + nParamList[idxm]/2 * log(2*pi) - 0.5 * log(det(H)) 

print(sprintf("Estimated value: %.2f", paramest))
print(sprintf("log marginal likelihood: %.2f", lml[idxm]))

# laplace
laplace <- function(lpMAP, mu, muMAP, H) {
  return(lpMAP - H/2 * (mu-muMAP)^2 )
}

lines(alphaL,exp(laplace(lp, alphaL, paramest[1], H[1])),
     type = "l", lty = "dashed", 
     lwd = 2
     )

legend("topright",
       lwd = c(2,2), 
       legend = c("exact", "Laplace"),
       lty=c("solid","dashed")# , 
       # bty = "n"
)

# 
# cat("----------- MAP -----------\n")
# resultMAP <- paramfitMAP(modelfunctions, data, nParamList, priorList)

