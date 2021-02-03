library(forecast)
library(parallel)
library(Mcomp)
library(tseries)
library(ForeCA)
library(forecTheta)
library(neuralnet)
library(nnet)
library(ggplot2)
library(GGally)
library(pracma)
library(e1071)
library(fracdiff)
library(tseriesChaos)
library(boot)
library(cramer)
library(alphahull)
library(entropy)
library(ggpubr)

tl <- function(x, ...){
  fit <- supsmu(1:length(x), x)
  out <- ts(cbind(trend=fit$y, remainder=x-fit$y))
  tsp(out) <- tsp(as.ts(x))
  return(structure(list(time.series=out),class="stl"))
}
CalStats2<-function(insample){
  
  Frequency<-frequency(insample)
  
  lambda<-BoxCox.lambda(insample, method=c("loglik"), lower=-1, upper=1)
  insample2<-BoxCox(insample,lambda)
  
  if (Frequency>1){
    decresults<-stl(insample2,s.window = "periodic")
    Seasonality <- decresults$time.series[,1]
    Randomness <- decresults$time.series[,3]
    Trend <- decresults$time.series[,2]
    
    Dec<-decompose(insample,type="additive")
    des<-insample-Dec$seasonal
    y<-des ; x<-(1:length(y)) ; tm<-lm(y~x)
    destrend<-des-predict(tm)
    
  }else{
    decresults<-tl(insample2)
    Seasonality <- decresults$time.series[,1]-decresults$time.series[,1]
    Randomness <- decresults$time.series[,2]
    Trend <- decresults$time.series[,1]
    
    des<-insample
    y<-des ; x<-(1:length(y)) ; tm<-lm(y~x)
    destrend<-des-predict(tm)
  }
  
  IndFrequency<-Frequency
  IndSlevel<-max((1-(var(Randomness)/var(insample2-Trend))),0)
  IndTlevel<-max((1-(var(Randomness)/var(insample2-Seasonality))),0)
  IndRlevel<-spectral_entropy(insample)[1]
  IndBlevel<-BoxCox.lambda(insample, method=c("loglik"), lower=0, upper=1)
  IndAlevel<-abs(acf(destrend,plot=FALSE)$acf[2])
  IndSk<-abs(skewness(insample)[1])
  IndKu<-min(kurtosis(insample)[1],7)
  IndNl<-terasvirta.test(insample)$p.value
  IndHu<-fracdiff(insample)$d+0.5
  
  return(c(IndRlevel,IndTlevel,IndSlevel,IndFrequency,IndAlevel,IndBlevel,IndSk,IndKu,IndNl,IndHu))
  
}

# PARAMETERS
setwd("/tmp")
in_file_path = 'series.csv'
out_file_path = 'output_feats.csv'

# Load data
Dw1 <- read.csv2(in_file_path, stringsAsFactors = F, sep=",", header = F)
data_form <- Dw1

colnames(data_form) <- c(paste0("X",seq(1:24)))
Results_data_form <- data.frame(matrix(NA,ncol=10,nrow=nrow(data_form)))
colnames(Results_data_form)<-c("Rlevel","Tlevel","Slevel",
                               "Frequency","Alevel","Blevel",
                               "Sklevel","Kulevel","Nllevel",
                               "SSlevel")

min_val <- 2000
max_val <- 5000
for (tsid in 1:nrow(data_form)){
  temp <- ts(as.numeric(data_form[tsid,]), frequency = 1)
  temp <- temp*(max_val - min_val) + min_val
  
  t <- try(CalStats2(temp))
  if("try-error" %in% class(t)){
    t <- rep(NA,10)
    } 
  
  Results_data_form[tsid,] <- t
  }

# save.image(file="GAN/ThanosFeats.RData")
write.csv(Results_data_form, out_file_path,row.names=F)

