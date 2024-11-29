#clear Rstudio
rm(list=ls()) #remove the variables in the environment 
options(scipen=999) #remove scientific notation

# Get the required libraries 
library(reticulate) # for processing python data types
library(dplyr)# for data manipulation 
library(tidyverse) # exponential smoothing via state space models and automatic ARIMA modelling.
library(zoo) # for creating time series
library(forecast) # for modelling packages 
library(ggplot2) # for plots 
library(fpp2) 

# Import the data 
pd <- import("pandas")
x <- pd$read_pickle("df_timeseries.pickle")
df <- as.data.frame(x)
head(df)

# group the data frame by probe_id and  and destination 
library(dplyr)
df_grouped <- df %>% 
  group_by('nprb_id','dst_addr')

# filter the grouped data frame based on two conditions
df_filtered <- df_grouped %>% 
  filter(nprb_id == "es1" & dst_addr == "52.46.200.93")
  
print(df_filtered)

df_avg <- df_filtered[c("new_time","avg")]
df_avg

# Create a time series object
# In R, the frequency is the number of observations in one "period". 
avg <- ts(df_avg$avg, frequency = 141)
str(avg)

#time series decomposition
#Decompose the time series into trend, seasonal, and remainder components
decomp<-stl(avg,s.window="periodic")  

# Plot the decomposition results
plot(decomp)

#naive forecast 
naive_forecast <- c(NA, avg[-length(avg)])
naive_forecast

df_avg$naive_forecast <- naive_forecast
df_avg$naive_forecast <- na.fill(df_avg$naive_forecast, 0)
head(df_avg)

#plot the naive forecast 
ggplot(df_avg, aes(x = new_time)) +
  geom_line(aes(y = avg , color = "df_avg")) +
  geom_line(aes(y = naive_forecast, color = "naive_forecast"))+
  scale_color_brewer(palette="Set1")

#exponential smoothing 
#Apply simple exponential smoothing with alpha = 0.2 && alpha = 0.8
ses_model_0.2 <- ses(avg, alpha = 0.2)
ses_model_0.8 <- ses(avg, alpha = 0.8)

# Extract the in-sample fitted values
ses_forecast_0.2 <- fitted(ses_model_0.2)
ses_forecast_0.8 <- fitted(ses_model_0.8)

df_avg$ses_forecast_0.2 <- ses_forecast_0.2
df_avg$ses_forecast_0.8 <- ses_forecast_0.8

#lets look at the updated data frame 
df_avg

#plot the simple exp smoothing forecast 
p <-ggplot(df_avg, aes(x = new_time)) +
  geom_line(aes(y = avg , color = "real_roundtriptime")) +
  geom_line(aes(y = ses_forecast_0.2, color = "simple-smoothing_0.2"))+
  geom_line(aes(y = ses_forecast_0.8, color = "simple-smoothing_0.8"))+
  scale_color_brewer(palette="Set2")

p + theme(legend.text = element_text(size = 16, family = "Helvetica Neue"))
# arima model 
# Needed libraries
library(tseries) # Load the tseries package
library(vrtest) # Load the vrtest package

#check for stationarity of variance and mean 
adf.test(avg)
Auto.VR(avg)
#df1_avg <- diff(avg)

#finding the AR and MA values 
acf(avg,main="acf of rtt values",lag.max=200)
pacf(avg,main="pacf of rtt values",lag.max=200)

#estimating the arima model
arima_model <- auto.arima(avg)
summary(arima_model)
tsdiag(arima_model)
checkresiduals(arima_model)
autoplot(arima_model)

arimar <- arima_model$residuals
ggtsdisplay(arimar,main="arima residuals")

arima_forecast <- fitted(arima_model)
df_avg$arima <- arima_forecast

ggplot(df_avg, aes(x = new_time)) +
  geom_line(aes(y = avg , color = "df_avg")) +
  geom_line(aes(y = arima , color = "arima")) +
  scale_color_brewer(palette="Set2")

# Holt Linear Method 
hw_model <- holt(avg,alpha = 0.8,beta = 0.2)

# Generate an in-sample forecast for your data
hw_forecast <- fitted(hw_model)

# View the forecast values
hw_forecast
df_avg$hw_forecast <- hw_forecast

# Plot the samples and fitted values
autoplot(avg) +  
  autolayer(hw_forecast, series = "hw_forecast") 

#forecasting using facebook prophet 
library(prophet)

#modify the data frame for the prophet algorithm
prophet_df <- df_avg %>% 
  rename(ds = new_time, y = avg)%>% 
  select(ds, y)

#create the model 
prophet_model <- prophet(prophet_df)

# Make in-sample predictions for the entire data set
prophet_forecast <- predict(prophet_model)
prophet_forecast_values <- prophet_forecast$yhat

#adding the prophet forecast values to original data frame 
df_avg$prophet_values<- prophet_forecast_values

#Visualizing the trend of the prophet forecast 
ggplot(df_avg, aes(x = new_time)) +
  geom_line(aes(y = avg , color = "avg")) +
  geom_line(aes(y = prophet_values, color = "prophet_value"))+
  scale_color_brewer(palette="Set3")
  

#using the garch model 
library(rugarch) # for garch models

# Extract the ARMA order from the model
arma_order <- arimaorder(arima_model)

p <- arma_order[1]
d <- arma_order[2]
q <- arma_order[3]

# Fit the GARCH model
garch_spec <- ugarchspec(mean.model = list(armaOrder = c(p, q, d)))
garch_model <- ugarchfit(spec = garch_spec, data = avg)

#get a GARCH variance series
vol <- ts(garch_model@fit$sigma^2)
plot(vol)

# Extract the fitted values and remove the date-time column
garch_values <- as.numeric(fitted(garch_model))

# Combine the original data with the fitted values
df_avg$garch_values <- garch_values

#Visualizing the trend of the garch forecast 
ggplot(df_avg, aes(x = new_time)) +
  geom_line(aes(y = avg , color = "avg")) +
  geom_line(aes(y = garch_values, color = "garch_value"))

hist(df_grouped$avg,main="Round trip time distribution",freq=TRUE,col="blue",xlab="round trip time")
#using a dynamic linear model
library(MARSS)

#define the model list 
mod_list <- list(
  B="identity",
  U="zero",
  Q=matrix("q"),
  Z="identity",
  A=matrix("a"),
  R=matrix("r"))

#fit the model with MARSS 
mars_fit <- MARSS(avg,mod_list)

#check accuracy of parameter prediction
MARSSparamCIs(mars_fit)

# Generate in-sample forecasts
mars_forecast <-predict(mars_fit)

#plot the prediction
autoplot(mars_forecast, main = "Forecast with MARSS Package")

#add the dynamic linear model prediction values to the data frame 
df_avg$mars_forecast <- mars_forecast$pred$estimate

# tbats forecast
# Fit a TBATS model to the time series
#  tbats(
#   y,
#   use.box.cox = NULL,
#   use.trend = NULL,
#   use.damped.trend = NULL,
#   seasonal.periods = NULL,
#   use.arma.errors = TRUE,
#   use.parallel = length(y) > 1000,
#   num.cores = 2,
#   bc.lower = 0,
#   bc.upper = 1,
#   biasadj = FALSE,
#   model = NULL,
#   ...
# ) 
tbats_model <- tbats(avg)

# Generate an in-sample forecast for the next 5 observations
tbats_forecast <- forecast(tbats_model)

# View the forecast values
tbats_values <- tbats_forecast$fitted

#add the tbats prediction values to the data frame 
df_avg$tbats_values <- tbats_values

#plot the predicted values
ggplot(df_avg, aes(x = new_time)) +
  geom_line(aes(y = avg , color = "avg")) +
  geom_line(aes(y = tbats_values, color = "tbat_values"))+
  scale_color_brewer(palette="Set3")

# regression tree forecast 
# the libraries
library(rpart)
library(rpart.plot)
# split the data 
train <- df_grouped %>% filter(new_time < "2023-01-18")
test <- df_grouped %>% filter(new_time > "2023-01-18")
glimpse(train)

# fit a regression tree using one of the destinations 
decison_tree <- rpart( nprb_id  ~ avg + dst_addr  , data = train)
printcp(decison_tree)

# visualize the regression tree 
rpart.plot(decison_tree)

#Make predictions on the testing set: 
predictions <- predict(decison_tree, newdata = test)
compare <- data.frame(test$new_time,test$avg,predictions)
compare

#plot the predicted values
ggplot(compare, aes(x = test$new_time)) +
  geom_line(aes(y = test$avg , color = "avg")) +
  geom_line(aes(y = predictions[0:length(test$avg)], color = "predictions"))















