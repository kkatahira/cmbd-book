#---------------------------------------------------------------------------#
# 周辺尤度の説明のための対数事後確率密度 (ここでは対数尤度と同じ)をプロット
#---------------------------------------------------------------------------#
rm(list=ls())
graphics.off()

source("model_functions_FQ.R")
source("parameter_fit_functions.R")

simulation_ID <- "FQlearning_siglesubject" # simulation ID for file name

csv_simulation_data <- paste0("./data/simulation_data", simulation_ID, ".csv")

dt <- read.table(csv_simulation_data, header = T, sep = ",")
data <- list(reward = dt$r, choice = dt$choice) 


# Contour map (DF-Q) ------------------------------------------------------

alphaL <- seq(0.01,0.99, by = 0.01)
alphaF <- seq(0.01,0.99, by = 0.01)
alphaL <- seq(0.0,1, by = 0.01)
alphaF <- seq(0.0,1, by = 0.01)

llval <- matrix(NA, nrow = length(alphaL), ncol = length(alphaF))

for (idxa in 1:length(alphaL)) {
  for (idxb in 1:length(alphaF)) {
    llval[idxa,idxb] <- - func_minimize(c(alphaL[idxa], alphaF[idxb], 2.0), 
                                        modelfunc = func_dfqlearning, 
                                        data = data, 
                                        prior = NULL #priorList[[3]]
                                        )
  }
}

x11(width = 15,height = 8)

par(mfcol=c(1,2))
par(oma=c(1.5,3,2,1))
par(mar = c(4, 4.5, 4.0, 2))

image(alphaL, alphaF, exp(llval),
      xlim = c(min(alphaL),max(alphaL)),
      ylim = c(min(alphaF),max(alphaF)),
      xlab = expression(
        paste("alphaL")
      ),
      ylab=expression(
        paste("alphaF")
      ),
      col = gray.colors(40),
      main = sprintf("DF-Q learning : log.ml = %.2f / max.ll = %.2f", 
                     mean(log(exp(llval))),
                     max(llval)),
      las = 1,
      cex      = 1.2,
      cex.lab  = 1.2,
      cex.axis = 1.2,
      cex.main = 1.2)

contour(alphaL, alphaF, exp(llval), nlevels = 5,
        xlim = c(min(alphaL),max(alphaL)),
        ylim = c(min(alphaF),max(alphaF)),
        xlab = expression(
          paste("alphaL")
        ),
        ylab=expression(
          paste("alphaF")
        ),
        las = 1,
        labcex = 1.1,
        add = T)

abline(a = 0, b = 1, lwd = 2)
box()

# plot likelihood (F-Q) ---------------------------------------------------

alphaL <- seq(0.0,1, by = 0.01)

llval <- numeric(length = length(alphaL))

for (idxa in 1:length(alphaL)) {
  llval[idxa] <- - func_minimize(c(alphaL[idxa], 2.0), 
                                      modelfunc = func_fqlearning, 
                                      data = data, 
                                      prior = NULL
  )
}

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
