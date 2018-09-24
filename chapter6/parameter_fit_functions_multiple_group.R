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
func_minimize_FE <- function(modelfunc, param, gpmap, data, prior)
{
  
  grouplist <- dplyr::distinct(data, group)$group
  
  totalnegll <- 0
  
  for (idxGroup in grouplist) {
    
    groupdata <- dplyr::filter(data, group == idxGroup)
    sublist <- dplyr::distinct(groupdata, subject)$subject
    
    for (idxsub in sublist) {
      
      subdata <- dplyr::filter(groupdata, subject == idxsub)
      
      ret <- modelfunc(param, gpmap, idxGroup, 
                       list(reward = subdata$reward, choice = subdata$choice), 
                       prior)
      totalnegll <- totalnegll + ret$negll
    }
  }
  
  # return total negative log-likelihood
  return(totalnegll)
}

#----------------------------------------------------------#
# Parameter fit fixed-effect ML
#----------------------------------------------------------#
paramfitFEML <- function(modelfunctions, gpmap, data)
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
    paramlist[[idxm]] <- numeric(max(gpmap[[idxm]]))
    
    fvalmin <- Inf
    for (idx in 1:10) {
      
      # set initial value
      initparam <- runif(max(gpmap[[idxm]]), 0, 1.0)
      
      res <- solnp(initparam, fun = func_minimize_FE, 
                   # LB = lblist[[idxmodel]], UB = ublist[[idxmodel]], 
                   control = list(trace = 0),
                   modelfunc = modelfunctions[[idxm]], 
                   gpmap = gpmap[[idxm]],
                   data = data, prior = NULL)
      
      nll <- res$values[length(res$values)]
      
      if (nll < fvalmin) {
        paramest <- res$par
        fvalmin <- nll
        
        res_ML_best <- res
      }
    }
    nTotalTrial <- dim(data)[1]
    aic[[idxm]]<- 2*fvalmin + 2 * max(gpmap[[idxm]])
    bic[[idxm]] <- 2*fvalmin + max(gpmap[[idxm]]) * log(nTotalTrial)
    negll[[idxm]] <- fvalmin
    paramlist[[idxm]] <- paramest
    
    print(sprintf("Estimated value: %.2f", paramest))
    print(sprintf("log-likelihood: %.2f, -AIC/2: %.2f, -BIC/2: %.2f", 
                  -negll[[idxm]], 
                  -aic[[idxm]]/2,
                  -bic[[idxm]]/2))
  }
  return(list(negll = negll, aic = aic, bic = bic, paramlist = paramlist))
}


# for check
# resultFEML <- paramfitFEML(modelfunctions, gpmap, data)

#----------------------------------------------------------#
# objective function to be minimized (for parameter fit)
#----------------------------------------------------------#
func_ML <- function(modelfunc, param, gpmap, idxGroup, data, prior)
{
  # return negative log-likelihood
  return(modelfunc(param, gpmap, idxGroup, data, prior)$negll)
}


#----------------------------------------------------------#
# Parameter fit single-subject ML
#----------------------------------------------------------#

fit_models_SSML <- function(data, modelfunctions, nParamList, lblist, ublist)
{
  
  ret <- list()
  p1est <- list()
  Qest <- list()
  paramest <- list()
  nModel <- length(modelfunctions)
  
  aic <- numeric(nModel)
  bic <- numeric(nModel)
  
  df_result <- data.frame()
  
  for (idxmodel in 1:nModel) {
    fvalmin = Inf;
    
    # ML
    for (idxrun in 1:5) {
      # set initial value
      initparam <- runif(nParamList[idxmodel], 0, 1.0)
      
      res <- solnp(initparam, fun = func_ML, 
                   modelfunc = modelfunctions[[idxmodel]], 
                   control = list(trace = 0),
                   gpmap = matrix(1:nParamList[idxmodel], nrow = 1),
                   idxGroup = 1, 
                   data = data, prior = NULL)
      
      nll <- res$values[length(res$values)]
      
      if (nll < fvalmin) {
        paramest[[idxmodel]] <- res$par
        fvalmin <- nll
        
        res_ML_best <- res
      }
    }
    
    aic[idxmodel] <- 2 * fvalmin + 2 * nParamList[idxmodel]
    bic[idxmodel] <- 2 * fvalmin + nParamList[idxmodel] * log(dim(data)[1])
    
    cat(sprintf("\nModel %d: ", idxmodel))
    cat(sprintf("param: %.4f ", paramest[[idxmodel]]))
    cat(sprintf(" AIC: %.2f \n", aic[idxmodel]))
    
    df_result <- df_result %>%
      bind_rows(
        data.frame(
          model = idxmodel, 
          ll = fvalmin, 
          aic = aic[idxmodel], 
          param = t(paramest[[idxmodel]]))
      )
  }
  return(df_result)
}

paramfitSSML <- function(modelfunctions, data, nParamList)
{
  df_model_result <- data %>%
    dplyr::group_by(group, subject) %>%
    dplyr::do(fit_models_SSML(., modelfunctions, nParamList, lblist, ublist))
  
  return(df_model_result)
}

# for check
#nParamList <- c(2)
#resultSSML <- paramfitSSML(modelfunctions[1], data, nParamList)
# mean(resultSSML$aic[[1]])

#----------------------------------------------------------#
# Parameter fit MAP
#----------------------------------------------------------#
fit_models_SSMAP <- function(data, modelfunctions, nParamList, priorList, 
                             lblist, ublist)
{
  ret <- list()
  p1est <- list()
  Qest <- list()
  paramest <- list()
  nModel <- length(modelfunctions)
  
  lml <- numeric(nModel)

  df_result <- data.frame()
  
  for (idxmodel in 1:nModel) {
    fvalmin = Inf;
    
    # ML
    for (idxrun in 1:5) {
      # set initial value
      initparam <- runif(nParamList[idxmodel], 0, 1.0)
      
      # print(priorList[[idxmodel]])
      
      res <- solnp(initparam, fun = func_ML, 
                   # LB = lblist[[idxmodel]], UB = ublist[[idxmodel]], 
                   modelfunc = modelfunctions[[idxmodel]], 
                   control = list(trace = 0),
                   gpmap = matrix(1:nParamList[idxmodel], nrow = 1),
                   idxGroup = 1, 
                   data = data, 
                   prior = priorList[[idxmodel]])
      
      nll <- res$values[length(res$values)]
      
      if (nll < fvalmin) {
        paramest[[idxmodel]] <- res$par
        fvalmin <- nll
        res_ML_best <- res
        fvalmin <- nll
        lp <- -nll
        H <- res$hessian
      }
    }
    
    # log marginal likelihood (Laplace)
    lml[idxmodel] <- lp + nParamList[idxmodel]/2 * log(2*pi) - 0.5 * log(det(H)) 
    
    # show(paramest[[idxmodel]]) 
    cat(sprintf("\nModel %d: ", idxmodel))
    cat(sprintf("param: %.4f ", paramest[[idxmodel]]))
    cat(sprintf(" lml: %.2f \n", lml[idxmodel]))
    
    df_result <- df_result %>%
      bind_rows(
        data.frame(
          model = idxmodel, 
          neglp = fvalmin, 
          lml = lml[idxmodel], 
          param = t(paramest[[idxmodel]]))
      )
  }
  return(df_result)
}

paramfitSSMAP <- function(modelfunctions, data, nParamList, priorList)
{
  df_model_result <- data %>%
    dplyr::group_by(group, subject) %>%
    dplyr::do(fit_models_SSMAP(., modelfunctions, nParamList, 
                               priorList, lblist, ublist))
  
  return(df_model_result)
}
# for check
# nParamList <- c(2)
# resultSSMAP <- paramfitSSMAP(modelfunctions[1], data, nParamList, priorList)

