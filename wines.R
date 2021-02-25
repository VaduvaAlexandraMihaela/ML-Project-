# a) Load libraries
library(ggplot2)
library(mlbench)
library(caret)
library(lattice)
library(e1071)
library(corrplot)
library(correlation)
library(randomForest)
library(rpart)
library(rpart.plot)

#load dataset
dataset <- wines


#Summarize the data
str(dataset)
summary(dataset)

#check if missing values exist in dataset
is.na(dataset)
anyNA(dataset)

#omit NA values
na.omit(dataset)
dataset <- na.omit(dataset)

#dimension of dataset 
dim(dataset)

#list types for each attribute 
sapply(dataset,class)

head(dataset, n=20)

#class distribution 
y <- dataset$quality
cbind(freq=table(y),percentage=prop.table(table(y))*100)

#Distribution of quality 
ggplot(data=dataset) + geom_bar(mapping = aes(x=quality))

#Univariate plots 
names(dataset) <-make.names(names(dataset))
ggplot(data = dataset, aes(x =fixed.acidity,y=quality)) + geom_point()
ggplot(data = dataset, aes(x = volatile.acidity, y = quality)) +
  geom_point()
ggplot(data = dataset, aes(x = citric.acid, y = quality)) +
  geom_point()
ggplot(data = dataset, aes(x = residual.sugar, y = quality)) +
  geom_point()
ggplot(data = dataset, aes(x = chlorides, y = quality)) +
  geom_point()
ggplot(data = dataset, aes(x = free.sulfur.dioxide, y = quality)) +
  geom_point()
ggplot(data = dataset, aes(x = total.sulfur.dioxide, y = quality)) +
  geom_point()
ggplot(data = dataset, aes(x = density, y = quality)) +
  geom_point()
ggplot(data = dataset, aes(x = pH, y = quality)) +
  geom_point()
ggplot(data = dataset, aes(x = sulphates, y = quality)) +
  geom_point()
ggplot(data = dataset, aes(x = alcohol, y = quality)) +
  geom_point()

# correlation plot
data1 <- dataset[, c( "fixed acidity", 
                  "volatile acidity", 
                  "citric acid", 
                  "residual sugar",
                  "chlorides",
                  "free sulfur dioxide",
                  "total sulfur dioxide","density","pH","sulphates",
                  "alcohol","quality") ]
data1 <- na.omit(data1)
cor(data1)
corrplot(cor(data1), method="number")


#bar plot 
par(mfrow=c(2,4))
for(i in 2:9) {
  counts <- table(dataset[,i])
  name <- names(dataset)[i]
  barplot(counts, main=name)
}

ggplot(dataset, aes(quality)) + geom_bar(fill="orange")

#point plot 
ggplot(dataset , aes(x = type, y =quality)) + geom_point()

# bar plot switch the x and y axes
ggplot(dataset, aes(x = type, y = quality)) +
  geom_col() + 
  coord_flip()

ggplot(dataset, aes(quality)) + 
  geom_bar(fill="orange")



#Classification

#classification of the wines into good, bad and normal 
dataset$taste <-ifelse(dataset$quality < 6, 'bad','good')
dataset$taste[dataset$quality == 6] <- 'normal'
dataset$taste <- as.factor(dataset$taste)

table(dataset$taste)

#splitting the dataset 
set.seed(7)
index <- sample(nrow(dataset), 0.8*nrow(dataset))
train <- dataset[index, ]
test <- dataset[-index, ]

#omit NA values for train and test 
na.omit(train)
na.omit(test)
train <- na.omit(train)

#normalize column names
names(train) <- make.names(names(train))
names(test) <- make.names(names(test))


#Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10)
metric <- "Accuracy"

# a) linear algorithms
#GLMNET
set.seed(7)
fit.glmnet <- train(taste~.-quality, data=train, method="glmnet", metric=metric, trControl=control)
# SVM
set.seed(7)
fit.svm <- train(taste~.-quality, data=train, method="svmRadial", metric=metric, trControl=control)
# b) nonlinear algorithms
#knn
set.seed(7)
fit.knn <- train(taste~.-quality, data=train, method="knn", metric=metric, trControl=control)
#randomForest 
set.seed(7)
fit.rf <- train(taste~.-quality, data=train, method="rf", metric=metric, trControl=control)
#Naive Bayes 
set.seed(7)
fit.nb <- train(taste~.-quality,data=train,method="nb",metric=metric, trControl=control)

#Compare algorithms 
transform_results <- resamples(list(GLMNET=fit.glmnet, SVM=fit.svm, KNN=fit.knn,NVM=fit.nb, RF =fit.rf))
summary(transform_results)
dotplot(transform_results)

#Evaluate Algorithms : with Feature Selection step 

#remove correlated attributes 
#find attributes that are highly corrected 
set.seed(7)
cutoff <- 0.70
correlations <- cor(train[,2:12])
highlyCorrelated <- findCorrelation(correlations, cutoff=cutoff)
for (value in highlyCorrelated) {
  print(names(train)[value])
}
#create a new dataset without highly corrected features
train_features <- train[,-highlyCorrelated]

#Run algorithms using 10-fold cross validation 
control <- trainControl(method="repeatedcv", number=10)
metric <- "Accuracy"

# a) linear algorithms
#GLMNET
set.seed(7)
fit.glmnet <- train(taste~.-quality, data=train_features, method="glmnet", metric=metric, trControl=control)
# SVM
set.seed(7)
fit.svm <- train(taste~.-quality, data=train_features, method="svmRadial", metric=metric, trControl=control)
# b) nonlinear algorithms
#knn
set.seed(7)
fit.knn <- train(taste~.-quality, data=train_features, method="knn", metric=metric, trControl=control)
#randomForest 
set.seed(7)
fit.rf <- train(taste~.-quality, data=train_features, method="rf", metric=metric, trControl=control)
#Naive Bayes 
set.seed(7)
fit.nb <- train(taste~.-quality,data=train_features,method="nb",metric=metric, trControl=control)

#Compare algorithms 
transform_results <- resamples(list(GLMNET=fit.glmnet, SVM=fit.svm, KNN=fit.knn,NVM=fit.nb, RF =fit.rf))
summary(transform_results)
dotplot(transform_results)

#Evaluate algorithms : with Box-Cox Transformation

#Run algorithms using 10-fold cross validation 
control <- trainControl(method="repeatedcv", number=10)
metric <- "Accuracy"


# a) linear algorithms
#GLMNET
set.seed(7)
fit.glmnet <- train(taste~.-quality, data=train, method="glmnet", metric=metric,preProc=c("center", "scale", "BoxCox"), trControl=control)
# SVM
set.seed(7)
fit.svm <- train(taste~.-quality, data=train, method="svmRadial", metric=metric,preProc=c("center", "scale", "BoxCox"), trControl=control)
# b) nonlinear algorithms
#knn
set.seed(7)
fit.knn <- train(taste~.-quality, data=train, method="knn", metric=metric,preProc=c("center", "scale", "BoxCox"), trControl=control)
#randomForest 
set.seed(7)
fit.rf <- train(taste~.-quality, data=train, method="rf", metric=metric,preProc=c("center", "scale", "BoxCox"), trControl=control)
#Naive Bayes 
set.seed(7)
fit.nb <- train(taste~.-quality,data=train,method="nb",metric=metric,preProc=c("center", "scale", "BoxCox"), trControl=control)

#Compare algorithms 
transform_results <- resamples(list(GLMNET=fit.glmnet, SVM=fit.svm, KNN=fit.knn,NVM=fit.nb, RF =fit.rf))
summary(transform_results)
dotplot(transform_results)


print(fit.rf)


x <- test[,1:13]
y <- test[,14]

predictions <- predict(fit.rf, newdata=x)
print(predictions)


#calculate Accuracy 
table(predictions, test$taste)
(373+163+431) / nrow(test)

#save the model to disk 
saveRDS(fit.rf,"MyFinalModel.rds")

#use the model for prediction
print("load the model")
model <- readRDS("MyFinalModel.rds")

finalPredictions <- predict(model, x)
print(finalPredictions)
table(finalPredictions,test$taste)
(519+233+652)/ nrow(test)

#confusionMatrix
confusionMatrix(finalPredictions, test$taste)