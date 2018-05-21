library(readr)
library(caTools)
library(mlbench)
library(caret)
library(dummies)
library(Matrix)
#install.packages('mlbench')

train <- read_csv("E:/Projects/McKinsey/train.csv")
train_dataset = as.data.frame(train)

#Fill missing values in BMI
median(train_dataset$bmi,na.rm = T)
mean(train_dataset$bmi,na.rm = T)
plot(train_dataset$bmi,type = 'l')
train_dataset$bmi[!is.na(train_dataset$bmi) & train_dataset$bmi > 60]= NA
train_dataset$bmi[is.na(train_dataset$bmi)] = median(train_dataset$bmi,na.rm = T)
plot(train_dataset$bmi,type = 'l')

#Fill missing values in Smoking Status
unique(train_dataset$smoking_status)
train_dataset$smoking_status[is.na(train_dataset$smoking_status)] = 'No Information'
unique(train_dataset$smoking_status)

#Bin Age
train_dataset$Age_bin = floor(train_dataset$age/10)
unique(train_dataset$Age_bin)

colnames(train_dataset)
#convert to factors
train_dataset[c(2,4,5,6,7,8,11,13)] = lapply(train_dataset[c(2,4,5,6,7,8,11,13)],factor)

dummies = dummy.data.frame(train_dataset[c(2,4,5,6,7,8,11,13)])
dummy_dataset = data.frame(train_dataset[c(1,12)],dummies,train_dataset[c(9,10)])
colnames(dummy_dataset)

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# run the RFE algorithm
results <- rfe(x = dummy_dataset[,3:33],y = dummy_dataset[,2],rfeControl=control)

# summarize the results
print(results)

dummy_dataset = data.frame(dummy_dataset$id,dummy_dataset[results$optVariables],dummy_dataset$stroke)
colnames(dummy_dataset) = c('id',results$optVariables,'stroke')
#Split dataset
set.seed(1211)
split = sample.split(dummy_dataset$stroke, SplitRatio = 0.8)
training_set = subset(dummy_dataset, split == TRUE)
validation_set = subset(dummy_dataset, split == FALSE)

#using one hot encoding 
train_labels = training_set$stroke 
val_label = validation_set$stroke
training_data = model.matrix(~.+0,data = training_set[,c(1:9)]) 
Validation_data = model.matrix(~.+0,data = validation_set[,c(1:9)])

#train_labels = as.numeric(train_labels)-1
#val_label = as.numeric(val_label)-1

library(xgboost)

#preparing matrix 
dtrain = xgb.DMatrix(data = training_data,label = train_labels) 
dtest = xgb.DMatrix(data = Validation_data,label=val_label)

#default parameters
params = list(booster = "gbtree", objective = "binary:logistic", eta=0.3, 
              gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 30, nfold = 10,
                 showsd = T, stratified = T,metrics = 'auc' ,print_every_n = 10, early_stop_round = 10,
                 maximize = T)


xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 30, watchlist = list(val=dtest,train=dtrain), print_every_n = 10, early_stop_round = 10, maximize = F , eval_metric = "error")
#model prediction
xgbpred <- predict (xgb1,dtrain)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#confusion matrix
library(caret)
confusionMatrix (xgbpred, train_labels)

#model prediction
xgbpred_test <- predict (xgb1,dtest)
xgbpred_test <- ifelse (xgbpred_test > 0.5,1,0)

#confusion matrix
library(caret)
confusionMatrix (xgbpred_val, val_label)

library(pROC)
roc_obj <- roc(xgbpred_test,val_label)
auc(roc_obj)