library(FactoMineR)
library(factoextra)
library(mda)
library(MASS)
library(sjPlot)
library(rmutil)
library(effects)
library(lmtest)
library(survey)
library(caret)
library(e1071)
library(openxlsx)

thedata = read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", header = TRUE)
thedata <- na.omit(thedata)
demodata <- thedata[, c(2,3,4,5,17,19,21)]
demodata$gender <- as.character(demodata$gender)
demodata$Partner <- as.character(demodata$Partner)
demodata$Dependents <- as.character(demodata$Dependents)
demodata$PaperlessBilling <- as.character(demodata$PaperlessBilling)
demodata$Churn <- as.character(demodata$Churn)
demodata$gender[demodata$gender == "Female"] <- 0
demodata$gender[demodata$gender == "Male"] <- 1
demodata$Partner[demodata$Partner == "No"] <- 0
demodata$Partner[demodata$Partner == "Yes"] <- 1
demodata$Dependents[demodata$Dependents == "No"] <- 0
demodata$Dependents[demodata$Dependents == "Yes"] <- 1
demodata$PaperlessBilling[demodata$PaperlessBilling == "No"] <- 0
demodata$PaperlessBilling[demodata$PaperlessBilling == "Yes"] <- 1
demodata$Churn[demodata$Churn == "No"] <- 0
demodata$Churn[demodata$Churn == "Yes"] <- 1
demodata$gender <- as.numeric(demodata$gender)
demodata$Partner <- as.numeric(demodata$Partner)
demodata$Dependents <- as.numeric(demodata$Dependents)
demodata$PaperlessBilling <- as.numeric(demodata$PaperlessBilling)
demodata$Churn <- as.numeric(demodata$Churn)
demofactors <- demodata
demofactors$gender <- as.factor(demofactors$gender)
demofactors$Partner <- as.factor(demofactors$Partner)
demofactors$Dependents <- as.factor(demofactors$Dependents)
demofactors$PaperlessBilling <- as.factor(demofactors$PaperlessBilling)
demofactors$Churn <- as.factor(demofactors$Churn)
demofactors$SeniorCitizen <- as.factor(demofactors$SeniorCitizen)

cor(demodata)


                                                              
normal <- fitdistr(demodata$MonthlyCharges, 'normal')$loglik                                                         
logistic <- fitdistr(demodata$MonthlyCharges, 'logistic')$loglik                                                       
weibull <- fitdistr(demodata$MonthlyCharges, 'weibull')$loglik                                                        
lognormal <- fitdistr(demodata$MonthlyCharges, 'lognormal')$loglik                                                      
exponential <- fitdistr(demodata$MonthlyCharges, 'exponential')$loglik   
cauchy <- fitdistr(demodata$MonthlyCharges, 'cauchy')$loglik 

stat.desc(demodata)

PCAA <- PCA(demodata)
PCAA$var$cor
fvis_pca_var(PCAA)
MCAA <- MCA(demofactors)

linearmodel <- glm(formula = Churn ~ SeniorCitizen, data = demodata)
plot(predictorEffects(linearmodel))



multimodel <- lm(formula = cbind(MonthlyCharges, Churn, PaperlessBilling) ~ SeniorCitizen + gender + Dependents + Partner, family=binomial(link='logit'), data = demodata)
plot(predictorEffects(multimodel))

multimonth <- lm(formula = MonthlyCharges ~ SeniorCitizen + Dependents + Partner,data = demodata)
plot(multimonth)



train <-  thedata[sample(nrow(demodata), 500), ]
waldtrain <- lm(formula = cbind(MonthlyCharges, Churn, PaperlessBilling) ~ SeniorCitizen + gender + Dependents + Partner, family=binomial(link='logit'), data = train)
waldtraintwo <- lm(formula = cbind(MonthlyCharges, Churn, PaperlessBilling) ~ SeniorCitizen + gender + Dependents + Partner, family=binomial(link='logit'), data = train)
regTermTest(waldtrain,"Partner")
regTermTest(waldtrain,"Dependents")





monthlysenior <- lm(formula = MonthlyCharges ~ SeniorCitizen, data = demodata)
mseniorplot <- plot(predictorEffects(monthlysenior))

monthlydep <- lm(formula = MonthlyCharges ~ Dependents, data = demodata)
mdepplot <- plot(predictorEffects(monthlydep))

monthlypart <- lm(formula = MonthlyCharges ~ Partner, data = demodata)
mpartplot <- plot(predictorEffects(monthlypart))

monthlythree <- lm(formula = MonthlyCharges ~ Partner + SeniorCitizen + Dependents, data = demodata)
plot(predictorEffects(monthlythree))


monthlyall <- lm(formula = MonthlyCharges ~ Dependents*SeniorCitizen, data = demodata)
mallplot <- plot(predictorEffects(monthlyall))
regTermTest(monthlyall,"Dependents")

plot_model(monthlyall, type = "pred", terms = c("Dependents", "SeniorCitizen"))
plot_model(monthlyall, type = "pred", terms = c("Partner", "SeniorCitizen"))

multimanova <- manova(formula = cbind(MonthlyCharges, Churn, PaperlessBilling) ~ SeniorCitizen + gender + Dependents + Partner, data = demodata)



crossvaltrain <-  thedata[sample(nrow(demodata), 200), ]

controltrain <- trainControl(method = "repeatedcv", number = 200, repeats= 5)
crosstest <- train(MonthlyCharges ~ SeniorCitizen,  data=demodata, method="lm",trControl = controltrain)
crossval = predict(crosstest, newdata=crossvaltrain, interval = "prediction")

summary(crossvaltrain$MonthlyCharges)
summary(crossval)
plot(crossval ~ crossvaltrain$SeniorCitizen)
abline(lm(crossval ~ crossvaltrain$SeniorCitizen))


distPlots <- function (x){
  mat <- rbind(1:3)
  layout(mat)
  hist(x, main="Frequency")
  boxplot(x, main='Boxplot')
  x.density <- density(x)
  plot(x.density, main="Density")
  polygon(x.density, col="green", border="blue")
  layout(c(1,1))
}

write.xlsx(demodata, file = "cleaneddata.xlsx")