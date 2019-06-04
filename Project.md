Practical Machine Learning Course Project
================
L. Brdar

## Overview

### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

### Data

The training data for this project are available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source:
<http://groupware.les.inf.puc-rio.br/har>. If you use the document you
create for this class for any purpose please cite them as they have been
very generous in allowing their data to be used for this kind of
assignment.

### The submission

The goal of this project is to build prediction model for the manner in
which they did the exercise. This is the “classe” variable in the
training set. The output of the project is going to be a report
describing how the model has been built, how was the cross validation
used, the expected out of sample error, and to explain the choices.
Finally, the prediction model will be used to predict 20 different test
cases (provided).

## Getting and Cleaning Data

Loading of necessary packages for the analysis and obtaining of data.
Data will be stored under ‘train’ and ‘test’, and missing values are
labeled as ‘NA’.

``` r
library(caret)
library(randomForest)
library(rpart)
set.seed(123)
train <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings=c("NA", "", "#DIV/0!"), stringsAsFactors = TRUE)
test <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings=c("NA", "", "#DIV/0!"), stringsAsFactors = TRUE)
```

We can get an overview of the data:

``` r
dim(train)
```

    ## [1] 19622   160

``` r
dim(test)
```

    ## [1]  20 160

The first, index column, and the timestamp columns are not relevant, so
we can exclude them. Furthermore, since the exploratory data showed that
many of the columns are almost entirely filled with NAs (sometimes
almost 100%), we can list those columns and exclude them also from
further
    analysis.

``` r
colMeans((is.na(train)))[colMeans((is.na(train)))>0.8]
```

    ##       kurtosis_roll_belt      kurtosis_picth_belt        kurtosis_yaw_belt 
    ##                0.9798186                0.9809398                1.0000000 
    ##       skewness_roll_belt     skewness_roll_belt.1        skewness_yaw_belt 
    ##                0.9797676                0.9809398                1.0000000 
    ##            max_roll_belt           max_picth_belt             max_yaw_belt 
    ##                0.9793089                0.9793089                0.9798186 
    ##            min_roll_belt           min_pitch_belt             min_yaw_belt 
    ##                0.9793089                0.9793089                0.9798186 
    ##      amplitude_roll_belt     amplitude_pitch_belt       amplitude_yaw_belt 
    ##                0.9793089                0.9793089                0.9798186 
    ##     var_total_accel_belt            avg_roll_belt         stddev_roll_belt 
    ##                0.9793089                0.9793089                0.9793089 
    ##            var_roll_belt           avg_pitch_belt        stddev_pitch_belt 
    ##                0.9793089                0.9793089                0.9793089 
    ##           var_pitch_belt             avg_yaw_belt          stddev_yaw_belt 
    ##                0.9793089                0.9793089                0.9793089 
    ##             var_yaw_belt            var_accel_arm             avg_roll_arm 
    ##                0.9793089                0.9793089                0.9793089 
    ##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
    ##                0.9793089                0.9793089                0.9793089 
    ##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
    ##                0.9793089                0.9793089                0.9793089 
    ##           stddev_yaw_arm              var_yaw_arm        kurtosis_roll_arm 
    ##                0.9793089                0.9793089                0.9832841 
    ##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
    ##                0.9833860                0.9798695                0.9832331 
    ##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
    ##                0.9833860                0.9798695                0.9793089 
    ##            max_picth_arm              max_yaw_arm             min_roll_arm 
    ##                0.9793089                0.9793089                0.9793089 
    ##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
    ##                0.9793089                0.9793089                0.9793089 
    ##      amplitude_pitch_arm        amplitude_yaw_arm   kurtosis_roll_dumbbell 
    ##                0.9793089                0.9793089                0.9795638 
    ##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
    ##                0.9794109                1.0000000                0.9795128 
    ##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
    ##                0.9793599                1.0000000                0.9793089 
    ##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
    ##                0.9793089                0.9795638                0.9793089 
    ##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
    ##                0.9793089                0.9795638                0.9793089 
    ## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell       var_accel_dumbbell 
    ##                0.9793089                0.9795638                0.9793089 
    ##        avg_roll_dumbbell     stddev_roll_dumbbell        var_roll_dumbbell 
    ##                0.9793089                0.9793089                0.9793089 
    ##       avg_pitch_dumbbell    stddev_pitch_dumbbell       var_pitch_dumbbell 
    ##                0.9793089                0.9793089                0.9793089 
    ##         avg_yaw_dumbbell      stddev_yaw_dumbbell         var_yaw_dumbbell 
    ##                0.9793089                0.9793089                0.9793089 
    ##    kurtosis_roll_forearm   kurtosis_picth_forearm     kurtosis_yaw_forearm 
    ##                0.9835898                0.9836408                1.0000000 
    ##    skewness_roll_forearm   skewness_pitch_forearm     skewness_yaw_forearm 
    ##                0.9835389                0.9836408                1.0000000 
    ##         max_roll_forearm        max_picth_forearm          max_yaw_forearm 
    ##                0.9793089                0.9793089                0.9835898 
    ##         min_roll_forearm        min_pitch_forearm          min_yaw_forearm 
    ##                0.9793089                0.9793089                0.9835898 
    ##   amplitude_roll_forearm  amplitude_pitch_forearm    amplitude_yaw_forearm 
    ##                0.9793089                0.9793089                0.9835898 
    ##        var_accel_forearm         avg_roll_forearm      stddev_roll_forearm 
    ##                0.9793089                0.9793089                0.9793089 
    ##         var_roll_forearm        avg_pitch_forearm     stddev_pitch_forearm 
    ##                0.9793089                0.9793089                0.9793089 
    ##        var_pitch_forearm          avg_yaw_forearm       stddev_yaw_forearm 
    ##                0.9793089                0.9793089                0.9793089 
    ##          var_yaw_forearm 
    ##                0.9793089

``` r
train <- train[, colMeans(is.na(train))<0.8]
train <- train[, -(3:7)]
train <- train[, -1]
train <- train[, colMeans(is.na(train))<0.8]
```

In order to be in line, the test set also undertakes the same filters.

``` r
test <- test[,intersect(colnames(train), colnames(test))]
```

### Partitioning

After obtaining tidy data, train data is split into train data
(trainSet) with 80% of data, and the rest 20% to the test data
(testSet).

``` r
partit <- createDataPartition(train$classe, p=0.8, list=FALSE)
trainSet <- train[partit, ]
testSet <- train[-partit, ]
```

## Decision Tree

The first testing model is the Decision tree from rpart package:

``` r
fit1 <- rpart(classe ~ ., data=trainSet, method="class")
predictionsDT <- predict(fit1, testSet, type = "class")
confm1 <- confusionMatrix(predictionsDT, testSet$classe)
confm1
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1014  150    8   68   24
    ##          B   28  437   60   28   49
    ##          C   23   75  555  105   90
    ##          D   34   68   37  402   44
    ##          E   17   29   24   40  514
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7448          
    ##                  95% CI : (0.7309, 0.7584)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6759          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9086   0.5758   0.8114   0.6252   0.7129
    ## Specificity            0.9109   0.9479   0.9095   0.9442   0.9656
    ## Pos Pred Value         0.8022   0.7259   0.6545   0.6872   0.8237
    ## Neg Pred Value         0.9616   0.9030   0.9580   0.9278   0.9373
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2585   0.1114   0.1415   0.1025   0.1310
    ## Detection Prevalence   0.3222   0.1535   0.2162   0.1491   0.1591
    ## Balanced Accuracy      0.9098   0.7618   0.8605   0.7847   0.8393

From the output it is possible to see that the accuracy of the model is
0.72, so there is room for improvement.

## Random Forest

The second model is Random Forest from randomForest package:

``` r
fit2 <- randomForest(classe ~ ., data=trainSet)
predictionRF <- predict(fit2, testSet, type = "class")
confm2 <- confusionMatrix(predictionRF, testSet$classe)
confm2
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    5    0    0    0
    ##          B    0  754    3    0    0
    ##          C    0    0  681    4    0
    ##          D    0    0    0  639    6
    ##          E    0    0    0    0  715
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9954          
    ##                  95% CI : (0.9928, 0.9973)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9942          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9934   0.9956   0.9938   0.9917
    ## Specificity            0.9982   0.9991   0.9988   0.9982   1.0000
    ## Pos Pred Value         0.9955   0.9960   0.9942   0.9907   1.0000
    ## Neg Pred Value         1.0000   0.9984   0.9991   0.9988   0.9981
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1922   0.1736   0.1629   0.1823
    ## Detection Prevalence   0.2858   0.1930   0.1746   0.1644   0.1823
    ## Balanced Accuracy      0.9991   0.9962   0.9972   0.9960   0.9958

The accuracy is now 0.9951, which is a significant improvement. The out
of sample error would then be 0.0049 (1-accuracy).

## Prediction of test data

Finally, the predicted values are:

``` r
predict <- predict(fit2, test, type = "class")
predict
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
