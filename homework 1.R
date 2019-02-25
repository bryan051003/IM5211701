library(ggplot2)
library(randomForest)
library(dplyr)
library(zoo)
library(ggcorrplot)
library(mice)
library(VIM)
library(missForest)
library(caret)


train<-read.csv("C:/Users/USER/Desktop/百日/chou/titanic/train.csv")
test<-read.csv("C:/Users/USER/Desktop/百日/chou/titanic/test.csv")
t<-bind_rows(train,test)

# 將空字串設定為 NA
t$Cabin<-as.character(t$Cabin)
t$Cabin[t$Cabin==""] <- "x"
t$C <- as.factor(sapply(t$Cabin, function(x) substr(x, 1, 1)))
t$Cabin<-as.factor(t$Cabin)
t$Embarked[t$Embarked==""] <- NA
t$Embarked <- as.factor(t$Embarked)
t$Pclass <- as.factor(t$Pclass)

summary(t)

md.pattern(t)

apply(t, 2, function(x) length(unique(x)))

#分析姓名，提取名称抬头
t$Title <- gsub("(.*, )|(\\..*)","",t$Name)
#名称抬头按性别列表显示
table(t$Sex, t$Title)
t$Title <- as.factor(t$Title)

t$Fsize <- t$SibSp + t$Parch + 1

tick.count<-aggregate(t$Ticket,by=list(t$Ticket),function(x) sum(!is.na(x)))
t$TicketCount <- apply(t, 1, function(x) tick.count[which(tick.count[, 1] ==x["Ticket"]),2])
t$TicketCount <- factor(sapply(t$TicketCount, function(x) ifelse(x > 1,'Share', 'Unique')))

### missForest ####
mtrain<-t[,-c(1,2,4,9,11)]
fit <- missForest(mtrain)
fit$OOBerror
t$Agena <- ifelse(is.na(t$Age),1,0)
t$Age <- fit$ximp$Age
t$Age <- round(t$Age)
t$Embarked <- fit$ximp$Embarked
t$Fare <- fit$ximp$Fare

#偏態
ggplot(t,aes(Fare))+geom_histogram()
t$Farelog<-log1p(t$Fare)
ggplot(t,aes(Farelog))+geom_histogram()

###target encoding
Sextm <- with(train, tapply(Survived, Sex, mean))
t$Sextm <- Sextm[match(t$Sex,names(Sextm))]

Tickettm <- with(train, tapply(Survived, Ticket, mean))
t$Tickettm <- Tickettm[match(t$Ticket,names(Tickettm))]
t$Tickettm <- ifelse(t$Tickettm==0,0,1)
t$Tickettm[is.na(t$Tickettm)] <- 0.5 #亂做

Pclasstm <- with(train, tapply(Survived, Pclass, mean))
t$Pclasstm <- Pclasstm[match(t$Pclass,names(Pclasstm))]

Embarkedtm <- with(train, tapply(Survived, Embarked, mean))
t$Embarkedtm <- Embarkedtm[match(t$Embarked,names(Embarkedtm))]

Ctm <- with(t[1:nrow(train),], tapply(Survived, C, mean))
t$Ctm <- Ctm[match(t$C,names(Ctm))]

Titletm <- with(t[1:nrow(train),], tapply(Survived, Title, mean))
t$Titletm <- Titletm[match(t$Title,names(Titletm))]
t$Titletm[is.na(t$Titletm)]<-0.5

#children teen
t$minor <- ifelse(t$Age<17,1,0)
t$minor <- as.factor(t$minor)

t$Survived <- as.factor(t$Survived)



tr<- t[,-c(1,4,9,11)]
ftest<- tr[892:nrow(t),]

summary(tr)
#5 fold cv
train_control <- trainControl(method="cv", number=5)

model1 <- train(Survived~., data=tr[1:nrow(train),], trControl=train_control, method="gbm")
model1
result1 <- predict(model1,ftest)
test$Survived<-result1
write.csv(test[,c(1,12)],file="C:/Users/USER/Desktop/百日//chou/titanic/result1.csv",row.names = FALSE)

model2 <- train(Survived~., data=tr[1:nrow(train),], trControl=train_control, method="rf")
model2
result2 <- predict(model2,ftest)
test$Survived<-result2
write.csv(test[,c(1,12)],file="C:/Users/USER/Desktop/百日//chou/titanic/result2.csv",row.names = FALSE)

#randomforest
set.seed(123)
rf_model = randomForest(Survived~.,data=tr[1:nrow(train),],ntree=500,mtry=8)
# 預測
result3 <- predict(rf_model,ftest)
test$Survived<-result3
write.csv(test[,c(1,12)],file="C:/Users/USER/Desktop/百日//chou/titanic/result3.csv",row.names = FALSE)


#tune ntree
# Observe that what is the best number of trees
plot(rf_model)

#tune mtry
set.seed(123)
tuneRF(tr[1:nrow(train),-1], tr[1:nrow(train),1])
# Variable Importance of Random Forest
rf_model$importance
varImpPlot(rf_model)
