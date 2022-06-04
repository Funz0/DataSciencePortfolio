#########################################Demo with a Possible final project dataset
# https://rpubs.com/H_Zhu/235617
#
library(ggplot2)
library(plyr)
library(ROCR)
adult <- read.table('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 
                    sep = ',', fill = F, strip.white = T)
colnames(adult) <- c('age', 'workclass', 'fnlwgt', 'educatoin', 
                     'educatoin_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 
                     'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income')
str(adult)
###three variables dropped
adult$educatoin <- NULL
adult$fnlwgt <- NULL
adult$relationship <- NULL

#Preprocessing/Cleaning/Exploring
# histogram of age by income group
ggplot(adult) + aes(x=as.numeric(age), group=income, fill=income) + 
  geom_histogram(binwidth=1, color='black')

# histogram of age by gender group
ggplot(adult) + aes(x=as.numeric(age), group=sex, fill=sex) + 
  geom_histogram(binwidth=1, color='black')

#The variable workclass stands for the industry in which the responding unit is employed.

summary(adult$workclass)

##Notice that there are two small groups - Never-worked and Without-pay. I will combine them with 
#Unknowns into a group called Other/Unknown. Those who work in the government are further break down 
#into federal, state, and local levels. To facilitate the analysis, I group them into one group called Government. While those who are self-employed fall into two groups, incorporated and not incorporated, and are grouped into  Self-Employed.

levels(adult$workclass)[1] <- 'Unknown'
# combine into Government job
adult$workclass <- gsub('^Federal-gov', 'Government', adult$workclass)
adult$workclass <- gsub('^Local-gov', 'Government', adult$workclass)
adult$workclass <- gsub('^State-gov', 'Government', adult$workclass) 

# combine into Sele-Employed job
adult$workclass <- gsub('^Self-emp-inc', 'Self-Employed', adult$workclass)
adult$workclass <- gsub('^Self-emp-not-inc', 'Self-Employed', adult$workclass)

# combine into Other/Unknown
adult$workclass <- gsub('^Never-worked', 'Other', adult$workclass)
adult$workclass <- gsub('^Without-pay', 'Other', adult$workclass)
adult$workclass <- gsub('^Other', 'Other/Unknown', adult$workclass)
adult$workclass <- gsub('^Unknown', 'Other/Unknown', adult$workclass)

adult$workclass <- as.factor(adult$workclass)
##
#Thus, there are now four categories for the variable workclass.

summary(adult$workclass)

##To explore the relationship between industry and income, the counts at the two income categories across 
#all four categories of workclass are calculated, as well as the in-group proportions. 
#The results are represented as a bar plot.

# barplot of job type by income group
# get the counts by industry and income group
count <- table(adult[adult$workclass == 'Government',]$income)["<=50K"]
count <- c(count, table(adult[adult$workclass == 'Government',]$income)[">50K"])
count <- c(count, table(adult[adult$workclass == 'Other/Unknown',]$income)["<=50K"])
count <- c(count, table(adult[adult$workclass == 'Other/Unknown',]$income)[">50K"])
count <- c(count, table(adult[adult$workclass == 'Private',]$income)["<=50K"])
count <- c(count, table(adult[adult$workclass == 'Private',]$income)[">50K"])
count <- c(count, table(adult[adult$workclass == 'Self-Employed',]$income)["<=50K"])
count <- c(count, table(adult[adult$workclass == 'Self-Employed',]$income)[">50K"])
count <- as.numeric(count)

# create a dataframe
industry <- rep(levels(adult$workclass), each = 2)
income <- rep(c('<=50K', '>50K'), 4)
df <- data.frame(industry, income, count)
df

## calculate the percentages
df <- ddply(df, .(industry), transform, percent = count/sum(count) * 100)

# format the labels and calculate their positions
df <- ddply(df, .(industry), transform, pos = (cumsum(count) - 0.5 * count))
df$label <- paste0(sprintf("%.0f", df$percent), "%")

# bar plot of counts by industry with in group proportions 
ggplot(df, aes(x = industry, y = count, fill = income)) +
  geom_bar(stat = "identity") +
  geom_text(aes(y = pos, label = label), size = 2) + 
  ggtitle('Income by Industry')

##Since education_num is a continuous representation of education, a stacked bar plot is used to visualize the relationship between  education_num and income, in-group proportions are calculated as well.

# create a dataframe
df1 <- data.frame(table(adult$income, adult$educatoin_num))
names(df1) <- c('income', 'education_num', 'count')
df1

# calculate the percentages
df1 <- ddply(df1, .(education_num), transform, percent = count/sum(count) * 100)

# format the labels and calculate their positions
df1 <- ddply(df1, .(education_num), transform, pos = (cumsum(count) - 0.5 * count))
df1$label <- paste0(sprintf("%.0f", df1$percent), "%")

# remove some in group percentage to avoid overlapped text
df1$label[which(df1$percent < 5)] <- NA

# bar plot of counts by years of education with in group proportions 
ggplot(df1, aes(x = education_num, y = count, fill = income)) +
  geom_bar(stat = "identity") +
  geom_text(aes(y = pos, label = label), size = 2) + 
  ggtitle('Income Level with Years of Education')

summary(adult$occupation)

levels(adult$occupation)[1] <- 'Unknown'
adult$occupation <- gsub('Adm-clerical', 'White-Collar', adult$occupation)
adult$occupation <- gsub('Craft-repair', 'Blue-Collar', adult$occupation)
adult$occupation <- gsub('Exec-managerial', 'White-Collar', adult$occupation)
adult$occupation <- gsub('Farming-fishing', 'Blue-Collar', adult$occupation)
adult$occupation <- gsub('Handlers-cleaners', 'Blue-Collar', adult$occupation)
adult$occupation <- gsub('Machine-op-inspct', 'Blue-Collar', adult$occupation)
adult$occupation <- gsub('Other-service', 'Service', adult$occupation)
adult$occupation <- gsub('Priv-house-serv', 'Service', adult$occupation)
adult$occupation <- gsub('Prof-specialty', 'Professional', adult$occupation)
adult$occupation <- gsub('Protective-serv', 'Service', adult$occupation)
adult$occupation <- gsub('Tech-support', 'Service', adult$occupation)
adult$occupation <- gsub('Transport-moving', 'Blue-Collar', adult$occupation)
adult$occupation <- gsub('Unknown', 'Other/Unknown', adult$occupation)
adult$occupation <- gsub('Armed-Forces', 'Other/Unknown', adult$occupation)
adult$occupation <- as.factor(adult$occupation)
summary(adult$occupation)

# create a dataframe
df2 <- data.frame(table(adult$income, adult$occupation))
names(df2) <- c('income', 'occupation', 'count')
df2

# calculate the percentages
df2 <- ddply(df2, .(occupation), transform, percent = count/sum(count) * 100)

# format the labels and calculate their positions
df2 <- ddply(df2, .(occupation), transform, pos = (cumsum(count) - 0.5 * count))
df2$label <- paste0(sprintf("%.0f", df2$percent), "%")

# bar plot of counts by occupation with in group proportions 
ggplot(df2, aes(x = occupation, y = count, fill = income)) +
  geom_bar(stat = "identity") +
  geom_text(aes(y = pos, label = label), size = 2) + 
  ggtitle('Income Level with Different Occupations')

summary(adult$marital_status)

adult$marital_status <- gsub('Married-AF-spouse', 'Married', adult$marital_status)
adult$marital_status <- gsub('Married-civ-spouse', 'Married', adult$marital_status)
adult$marital_status <- gsub('Married-spouse-absent', 'Married', adult$marital_status)
adult$marital_status <- gsub('Never-married', 'Single', adult$marital_status)
adult$marital_status <- as.factor(adult$marital_status)
summary(adult$marital_status)

df3 <- data.frame(table(adult$income, adult$marital_status))
names(df3) <- c('income', 'marital_status', 'count')
df3

# calculate the percentages
df3 <- ddply(df3, .(marital_status), transform, percent = count/sum(count) * 100)

# format the labels and calculate their positions
df3 <- ddply(df3, .(marital_status), transform, pos = (cumsum(count) - 0.5 * count))
df3$label <- paste0(sprintf("%.0f", df3$percent), "%")

# bar plot of counts by marital status with in group proportions 
ggplot(df3, aes(x = marital_status, y = count, fill = income)) +
  geom_bar(stat = "identity") +
  geom_text(aes(y = pos, label = label), size = 2) + 
  ggtitle('Income Level with Marital Status')

# histogram of capital_gain
ggplot(adult) + aes(x=as.numeric(capital_gain), group=income, fill=income) + 
  geom_histogram(bins=10, color='black') + ggtitle('Histogram of Capital Gain')

# histogram of capital_loss
ggplot(adult) + aes(x=as.numeric(capital_loss), group=income, fill=income) + 
  geom_histogram(bins=10, color='black') + ggtitle('Histogram of Capital Loss')

# percentage of observatiosn with no capital gain or loss
sum(adult$capital_gain == 0)/length(adult$capital_gain)

##drop skewed variables
adult$capital_gain <- NULL
adult$capital_loss <- NULL
adult$native_country <- NULL

##race
df4 <- data.frame(table(adult$income, adult$race))
names(df4) <- c('income', 'race', 'count')
df4

# calculate the percentages
df4 <- ddply(df4, .(race), transform, percent = count/sum(count) * 100)

# format the labels and calculate their positions
df4 <- ddply(df4, .(race), transform, pos = (cumsum(count) - 0.5 * count))
df4$label <- paste0(sprintf("%.0f", df4$percent), "%")

# do not display percentage for low counts categories
df4$label[df4$race == 'Other'] <- NA
df4$label[df4$race == 'Amer-Indian-Eskimo'] <- NA

# bar plot of counts by marital status with in group proportions 
ggplot(df4, aes(x = race, y = count, fill = income)) +
  geom_bar(stat = "identity") +
  geom_text(aes(y = pos, label = label), size = 2) + 
  ggtitle('Income Level by Race')

summary(adult)
library(summarytools)
b <-dfSummary(adult)
view(b)

write.csv(adult, "adult.clean.csv")
######################################################################3#######

#Model Fitting - Logistic Regression
#80% of the original data is used as the training set, while the rest 20% is used as test set.

sz <- round(.8 * dim(adult)[1])  # training set size
training_set <- adult[1:sz,]
testing_set <- adult[-(1:sz),]

m1 <- glm(income ~ ., data = training_set, family = binomial('logit'))
summary(m1)

#confidence interval
confint(m1)

# evaluation
m_full <- m1  # full model is the model just fitted
m_null <- glm(income ~ 1, data = training_set, family = binomial('logit'))

# backward selection
step(m_full, trace = F, scope = list(lower=formula(m_null), upper=formula(m_full)),
     direction = 'backward')
step(m1, direction="backward")

# forward selection
step(m_null, trace = F, scope = list(lower=formula(m_null), upper=formula(m_full)),
     direction = 'forward')
step(m1, direction="backward")


# create a data frame to store information regarding deviance residuals
index <- 1:dim(training_set)[1]
dev_resid <- residuals(m1)
income <- training_set$income
dff <- data.frame(index, dev_resid, income)

ggplot(dff, aes(x = index, y = dev_resid, color = income)) +
  geom_point() + 
  geom_hline(yintercept = 3, linetype = 'dashed', color = 'blue') +
  geom_hline(yintercept = -3, linetype = 'dashed', color = 'blue') +
  ggtitle('Plot of Deviance Residuals')


###prediction
prob <- predict(m1, testing_set, type = 'response')
pred <- rep('<=50K', length(prob))
pred[prob>=.5] <- '>50K'
# confusion matrix 
tb <- table(pred, testing_set$income)
tb

###Neural network##############do not run this block. Your computer may freeze.
library(nnet)
nn1 <- nnet(income ~ ., data = training_set, size = 40, maxit = 500)
nn1.pred <- predict(nn1, newdata = testing_set, type = 'raw')

pred1 <- rep('<=50K', length(nn1.pred))
pred1[nn1.pred>=.5] <- '>50K'
# confusion matrix 
tb1 <- table(pred1, testing_set$income)
tb1
# pred1   <=50K >50K
# <=50K  4494  678
# >50K    418  922
# 83% accuracy (4494+922)/(4494+922+678+418)


###

###CART model
library(rpart)
tree2 <- rpart(income ~ ., data = training_set, method = 'class', cp = 1e-3)
tree2.pred.prob <- predict(tree2, newdata = testing_set, type = 'prob')
tree2.pred <- predict(tree2, newdata = testing_set, type = 'class')
# confusion matrix 
tb2 <- table(tree2.pred, testing_set$income)
tb2
##accuracy is 83%

#####################randomForest
library(randomForest)
rf3 <- randomForest(income ~ ., data = training_set, ntree = 1000)
rf3.pred.prob <- predict(rf3, newdata = testing_set, type = 'prob')
rf3.pred <- predict(rf3, newdata = testing_set, type = 'class')
# confusion matrix 
tb3 <- table(rf3.pred, testing_set$income)
tb3
#The prediction result of RF has an accuracy of 83.65%, and a misclassification rate of 16.35%. 

###################SVM
library(kernlab)
svm4 <- ksvm(income ~ ., data = training_set)
svm4.pred.prob <- predict(svm4, newdata = testing_set, type = 'decision')
svm4.pred <- predict(svm4, newdata = testing_set, type = 'response')
# confusion matrix 
tb4 <- table(svm4.pred, testing_set$income)
tb4
##          
## svm4.pred <=50K >50K
##     <=50K  4582  773
##     >50K    330  827
#The prediction result of SVM has an accuracy of 83.06%, and a misclassification rate of 16.94%.

#Conclusion
#ROC curve is a plot of true positive rate against false positive rate under all threshold values. 
#Confusion matrix is a measurement of overall prediction accuracy. Since the majority of observations 
#in the data set has income less than $50,000 a year, sensitivity and specificity contribute to the overall accuracy by different weights. The five different classifiers are compared using ROC curve.

# create a prediction object
pr <- prediction(prob, testing_set$income)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")

# create a data frame for TP and FP rates
dd <- data.frame(FP = prf@x.values[[1]], TP = prf@y.values[[1]])

# NN
pr1 <- prediction(nn1.pred, testing_set$income)
prf1 <- performance(pr1, measure = "tpr", x.measure = "fpr")
dd1 <- data.frame(FP = prf1@x.values[[1]], TP = prf1@y.values[[1]])

# CART
pr2 <- prediction(tree2.pred.prob[,2], testing_set$income)
prf2 <- performance(pr2, measure = "tpr", x.measure = "fpr")
dd2 <- data.frame(FP = prf2@x.values[[1]], TP = prf2@y.values[[1]])

# RF
pr3 <- prediction(rf3.pred.prob[,2], testing_set$income)
prf3 <- performance(pr3, measure = "tpr", x.measure = "fpr")
dd3 <- data.frame(FP = prf3@x.values[[1]], TP = prf3@y.values[[1]])

# SVM
pr4 <- prediction(svm4.pred.prob, testing_set$income)
prf4 <- performance(pr4, measure = "tpr", x.measure = "fpr")
dd4 <- data.frame(FP = prf4@x.values[[1]], TP = prf4@y.values[[1]])



# plot ROC curve for logistic regression
g <- ggplot() + 
  geom_line(data = dd, aes(x = FP, y = TP, color = 'Logistic Regression')) + 
  geom_line(data = dd1, aes(x = FP, y = TP, color = 'Neural Networks')) + 
  geom_line(data = dd2, aes(x = FP, y = TP, color = 'CART')) + 
  geom_line(data = dd3, aes(x = FP, y = TP, color = 'Random Forest')) +
  geom_line(data = dd4, aes(x = FP, y = TP, color = 'Support Vector Machine')) +
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1)) +
  ggtitle('ROC Curve') + 
  labs(x = 'False Positive Rate', y = 'True Positive Rate') 


g +  scale_colour_manual(name = 'Classifier', values = c('Logistic Regression'='#E69F00', 
                                                         'Neural Networks'='#56B4E9', 'CART'='#009E73', 
                                                         'Random Forest'='#D55E00', 'Support Vector Machine'='#0072B2'))


####
# AUC
auc <- rbind(performance(pr, measure = 'auc')@y.values[[1]],
             performance(pr1, measure = 'auc')@y.values[[1]],
             performance(pr2, measure = 'auc')@y.values[[1]],
             performance(pr3, measure = 'auc')@y.values[[1]],
             performance(pr4, measure = 'auc')@y.values[[1]])
rownames(auc) <- (c('Logistic Regression', 'Neural Networks', 'CART',
                    'Random Forest', 'Support Vector Machine'))
colnames(auc) <- 'Area Under ROC Curve'
round(auc, 4)
##                        Area Under ROC Curve
## Logistic Regression                  0.8825
## Neural Networks                      0.8871
## CART                                 0.8501
## Random Forest                        0.8718
## Support Vector Machine               0.8608
###Logistic regression has the highest area under curve (AUC) value, while CART has the lowest AUC. Machine learning is no denying a powerful, but it should not be considered as a substitute of traditional statistical modeling.