library(tidyverse)
library(Rsolnp)

#----------------------------------------------------------#
# objective function to be minimized (for single subject parameter fit)
#----------------------------------------------------------#
func_minimize <- function(modelfunc, param, data, prior)
{
  ret <- modelfunc(param, data, prior)
  
  # return negative log-likelihood
  return(ret$negll)
}

#----------------------------------------------------------#
# objective function to be minimized (for fixed effect parameter fit)
#----------------------------------------------------------#
func_minimize_FE <- function(modelfunc, param, data, prior)
{
  sublist <- dplyr::distinct(data, subject)$subject
  nSubject <- length(sublist)
  
  totalnegll <- 0
  
  # cat(sublist)
  
  for (idxsub in sublist) {
    subdata <- dplyr::filter(data, subject == idxsub)
    
    ret <- modelfunc(param, list(reward = subdata$r, choice = subdata$choice), prior)
    totalnegll <- totalnegll + ret$negll
  }
  # return total negative log-likelihood
  return(totalnegll)
}

#----------------------------------------------------------#
# Parameter fit fixed-effect ML
#----------------------------------------------------------#
paramfitFEML <- function(modelfunctions, data, nParamList)
{
  nModel <- length(modelfunctions)
  
  aic <- list() # numeric(nModel)
  bic <- list() # numeric(nModel)
  negll <- list() # numeric(nModel)
  paramlist <- list()
  
  for (idxm in 1:nModel) {
    
    print(sprintf("Model %d:", idxm))
    
    aic[[idxm]] <- 0
    bic[[idxm]] <- 0
    negll[[idxm]] <- 0
    paramlist[[idxm]] <- numeric(nParamList[idxm])
    
    fvalmin <- Inf
    for (idx in 1:10) {
      
      # set initial value
      initparam <- runif(nParamList[idxm], 0, 1.0)
      
      res <- solnp(initparam, fun = func_minimize_FE, 
                   # LB = lblist[[idxmodel]], UB = ublist[[idxmodel]], 
                   modelfunc = modelfunctions[[idxm]], 
                   control = list(trace = 0),
                   data = data, prior = NULL)
      
      nll <- res$values[length(res$values)]
      
      if (nll < fvalmin) {
        paramest <- res$par
        fvalmin <- nll
        
        res_ML_best <- res
      }
    }
    T <- dim(data)[1]
    aic[[idxm]]<- 2*fvalmin + 2 * nParamList[idxm]
    bic[[idxm]] <- 2*fvalmin + nParamList[idxm] * log(T)
    negll[[idxm]] <- fvalmin
    paramlist[[idxm]] <- paramest
    
    print(sprintf("Estimated value: %.2f", paramest))
    print(sprintf("negative log-likelihood: %.2f, AIC: %.2f, BIC: %.2f", 
                  negll[[idxm]], 
                  aic[[idxm]],
                  bic[[idxm]]))
  }
  return(list(negll = negll, aic = aic, bic = bic, paramlist = paramlist))
}


# for check
# resultFEML <- paramfitFEML(modelfunctions, data, nParamList)


#----------------------------------------------------------#
# Parameter fit single-subject ML
#----------------------------------------------------------#
paramfitSSML <- function(modelfunctions, data, nParamList)
{
  nModel <- length(modelfunctions)
  sublist <- dplyr::distinct(data, subject)$subject
  
  aic <- list() # numeric(nModel)
  bic <- list() # numeric(nModel)
  negll <- list() # numeric(nModel)
  paramlist <- list()
  nSubject <- length(sublist)
  
  for (idxm in 1:nModel) {
    
    print(sprintf("Model %d:", idxm))
    
    aic[[idxm]] <- numeric(nSubject)
    bic[[idxm]] <- numeric(nSubject)
    negll[[idxm]] <- numeric(nSubject)
    paramlist[[idxm]] <- matrix(0, nSubject, nParamList[idxm])
    # print(paramlist)
    
    for (idxsub in sublist) {
      print(sprintf("Subject %d:", idxsub))
      subdata <- dplyr::filter(data, subject == idxsub)
      
      fvalmin <- Inf
      for (idx in 1:10) {
        
        # set initial value
        initparam <- runif(nParamList[idxm], 0, 1.0)
        
        res <- solnp(initparam, fun = func_minimize, 
                     # LB = lblist[[idxmodel]], UB = ublist[[idxmodel]], 
                     modelfunc = modelfunctions[[idxm]], 
                     control = list(trace = 0),
                     data = list(reward = subdata$r, choice = subdata$choice), prior = NULL)
        
        nll <- res$values[length(res$values)]
        
        if (nll < fvalmin) {
          paramest <- res$par
          fvalmin <- nll
          
          res_ML_best <- res
        }
      }
      T <- length(subdata$trial)
      aic[[idxm]][idxsub] <- 2*fvalmin + 2 * nParamList[idxm]
      print(sprintf("AIC %f:", aic[[idxm]][idxsub]))
      bic[[idxm]][idxsub] <- 2*fvalmin + nParamList[idxm] * log(T)
      negll[[idxm]][idxsub] <- fvalmin
      
      paramlist[[idxm]][idxsub,] <- paramest
      
      print(sprintf("Estimated value: %.2f", paramest))
      print(sprintf("log-likelihood: %.2f, AIC: %.2f, BIC: %.2f", 
                    negll[[idxm]][idxsub], 
                    aic[[idxm]][idxsub],
                    bic[[idxm]][idxsub]))
    }
  }
  return(list(negll = negll, aic = aic, bic = bic, paramlist = paramlist))
}

# for check
# resultSSML <- paramfitSSML(modelfunctions, data, nParamList)
# mean(resultSSML$aic[[1]])

#----------------------------------------------------------#
# Parameter fit MAP
#----------------------------------------------------------#
paramfitSSMAP <- function(modelfunctions, data,  nParamList, prior)
{
  
  nModel <- length(modelfunctions)
  sublist <- dplyr::distinct(data, subject)$subject
  
  nModel <- length(modelfunctions)
  lml <- list()
  neglp <- list()
  paramlist <- list()
  hessian <- list()
  nSubject <- length(sublist)
  
  for (idxm in 1:nModel) {
    
    lml[[idxm]] <- numeric(nSubject)
    neglp[[idxm]] <- numeric(nSubject)
    paramlist[[idxm]] <- matrix(0, nSubject, nParamList[idxm])

    for (idxsub in sublist) {
      print(sprintf("Subject %d:", idxsub))
      subdata <- dplyr::filter(data, subject == idxsub)
      
      
      fvalmin <- Inf
      print(sprintf("Model %d:", idxm))
      
      for (idx in 1:10) {
        
        # set initial value
        initparam <- runif(nParamList[idxm], 0, 1.0)
        
        res <- solnp(initparam, fun = func_minimize, 
                     # LB = lblist[[idxmodel]], UB = ublist[[idxmodel]], 
                     modelfunc = modelfunctions[[idxm]], 
                     control = list(trace = 0),
                     data = list(reward = subdata$r, choice = subdata$choice), 
                     prior = prior[[idxm]])
        
        nll <- res$values[length(res$values)]
        
        if (nll < fvalmin) {
          paramest <- res$par
          fvalmin <- nll
          lp <- -nll
          H <- res$hessian
          res_ML_best <- res
        }
      }
      
      T <- length(subdata$trial)
      neglp[[idxm]][idxsub] <- nll
      
      paramlist[[idxm]][idxsub,] <- paramest
      
      # log marginal likelihood (Laplace)
      lml[[idxm]][idxsub] <- lp + nParamList[idxm]/2 * log(2*pi) - 0.5 * log(det(H)) 
      
      print(sprintf("Estimated value: %.2f", paramest))
      print(sprintf("log marginal likelihood: %.2f", lml[[idxm]][idxsub]))
    }
    
  }
  return(list(neglp = neglp, lml = lml, paramlist = paramlist))
}

# for check
#resultSSMAP <- paramfitSSMAP(modelfunctions, data, nParamList, priorList)
#mean(resultSSMAP$lml[[1]])

