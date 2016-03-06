# Weight Lifting Exercise Quality Prediction
Steven Strand  
Friday, March 04, 2016  

## Overview

As part of a Coursera Class on Practical Machine Learning,
an analysis of dumbbell exercise data was performed using the caret package in R.
The analysis predicts the manner in which the participants did the exercise, based on a training set in which the quality was ranked with a classe variable ranging from A (good) to E (poor).
This report describes how the model was built, and how cross validation was used to improve the model. An estimate is made of the out of sample error based on cross validation results and predictions for a validation data set. The report describes the logic behind choosing the final model. The model was used to predict the classe variable for 20 test cases as part of a project quiz.

## Methodology
The following steps were used to build the final model:

### Pre-Processing

1) The data set was cleaned to remove blank and NA variables
1) A small number of outliers were removed. An outlier here is defined as points more than 10 times the IQR outside the IQR.
1) Some predictor variables such as time were excluded from the set of predictors

### Preliminary Analysis

1) Data were analyzed using exploratory plots and summaries

### Modeling with Various Learning Methods

1) The training data set was divided into a set for model training and another set for validation
1) Numerous learning methods in the caret package in R were used on the training set to build models
1) The models were evaluated on the validation set

### Optimization

1) The more promising learning methods were optimized by choosing additional pre-processing techniques 
1) Cross-validation was performed on some of the models
1) Various training set sizes were used to find a balance between computation time and accuracy
1) Parallel processing was used to speed up computations

### Application to the Test Data

1) The model was used to predict the classe variable for the test data
1) The test predictions were submitted on the course web site

## Result Summary

1) The best model used was a random forest model using 60% of the training data.
1) The random forest model was predicted internally to have an error rate of 0.5%
1) The error rate on the validation set was approximately 0.5 %
1) The error rate on the test set of 20 was 0%. All classe variables were predicted correctly using the random forest model.
1) The random forest model did not require any special pre-processing.
1) The time of computation for the random forest model was the longest of all those tested.

## Obtaining & Cleaning Data

### Reading in Data

```r
# Speed up computations with parallel processing
library(doParallel)
registerDoParallel(cores=3)

# Load the librararies
library(caret)
library(ggplot2)

# Read in the data set
pml.train <- read.csv("pml-training.csv",stringsAsFactors = FALSE)
pml.test <- read.csv("pml-testing.csv",stringsAsFactors = FALSE)
```

### Pre-Processing the Data


```r
# Find indices of columns with mostly NA values
n_na <- apply(pml.train,2,function(x) sum(is.na(x)))
fr_na <- n_na/nrow(pml.train)

idx_skip_na <- which(fr_na>0.97)
cat("There were ",length(idx_skip_na)," columns with more than 97% NA values\n")
```

```
## There were  67  columns with more than 97% NA values
```

```r
# Find indices of columns with mostly blank values
n_blank <- apply(pml.train,2,function(x) sum(x==""))
fr_blank <- n_blank/nrow(pml.train)
idx_skip_blank <- which(fr_blank>0.95)
cat("There were ",length(idx_skip_blank)," columns with more than 95% blank values\n")
```

```
## There were  33  columns with more than 95% blank values
```

```r
idx_skip <- c(idx_skip_na,idx_skip_blank)

training <- pml.train[,-idx_skip]
testing <- pml.test[,-idx_skip]

# Make certain variables into factors
training$classe <- as.factor(training$classe)
training$user_name <- as.factor(training$user_name)
testing$user_name <- as.factor(testing$user_name)

# Remove some variables from data frame that will not be used in the modeling
extra_vars <- c("X","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window")
training_sm <- training[,!names(training) %in% extra_vars]
testing_sm <- testing[,!names(testing) %in% extra_vars]

# Find any obviously bad data. 
# Here search for values more than a specified multiplier of the IQR in the training data

flag_outliers <- function(x,cutoff) {
    qnt <- quantile(x, probs=c(.25, .75))
    hdelta   <- cutoff* IQR(x)
    ilo <-which(x < (qnt[1]-hdelta))
    ihi <- which(x > (qnt[2]+hdelta))
    return(c(ilo,ihi))

}
check_outliers <- function(df1,cutoff) {
    idx_remove <- c()
    for( vname in colnames(df1)) {
        if(vname=="classe") next
        if(vname=="user_name") next
        iout <- flag_outliers(df1[[vname]],cutoff=cutoff)
        if(length(iout)>0) {
            cat("Outliers in Var:",vname,"\n")
            print(iout)
            idx_remove <- c(idx_remove,iout)
        }
    }
    return(unique(idx_remove))
}

# Look for bad data
idx_remove <- check_outliers(training_sm,cutoff=10)
```

```
## Outliers in Var: gyros_belt_x 
##  [1] 16016 16017 16018 16019 16020 17962 17963 17964 17965 17966
## Outliers in Var: gyros_dumbbell_x 
## [1] 5373
## Outliers in Var: gyros_dumbbell_y 
## [1]  152 5373
## Outliers in Var: gyros_dumbbell_z 
## [1] 5373
## Outliers in Var: magnet_dumbbell_y 
## [1] 9274
## Outliers in Var: gyros_forearm_x 
## [1] 5373
## Outliers in Var: gyros_forearm_y 
## [1] 5373
## Outliers in Var: gyros_forearm_z 
## [1]  941  947  952 5373
```

```r
cat("Rows in training data with outliers:", idx_remove,"\n")
```

```
## Rows in training data with outliers: 16016 16017 16018 16019 16020 17962 17963 17964 17965 17966 5373 152 9274 941 947 952
```

```r
# There are a small number of outliers in a large data set. It seems expedient to remove them.
training_sm2 <- training_sm[-idx_remove,]
```

### Split the Data

```r
# Create a subset for training and another for validation
set.seed(10101)

pfrac=0.1

inTrain <- createDataPartition(y=training_sm2$classe,p=pfrac,list=FALSE)
df_training <- training_sm2[inTrain,]
df_validation <- training_sm2[-inTrain,]
```

## Exploratory Data Analysis
Plots and summaries of the training data were used to identify issues such as data outliers, missing data and highly correlated variables. 
Some response variables were observed to have dramatically different mean values for different user_name values, especially for some variables associated with the belt measurements. 

The following plot shows histograms of the pitch_forearm variable data for various classes and user_names. The data for this variable are centered near each other for all users.


```r
ggplot() + facet_wrap(~classe)+geom_histogram(data=df_training, mapping=aes(x=pitch_forearm, fill=user_name))
```

![](strand-practical-machine-learning-assignment_files/figure-html/unnamed-chunk-4-1.png) 

The following histograms show that the roll_belt data is bimodal and unlike the data for pitch_forearm. In this case, the responses for some users are distinct from others. This bimodal feature should make training a learning model more difficult since the responses and variation by classe are different based on the user_name variable. This data feature suggests that creating individual models for each user might yield better predictions.


```r
ggplot() + facet_wrap(~classe)+geom_histogram(data=df_training, mapping=aes(x=roll_belt, fill=user_name))
```

![](strand-practical-machine-learning-assignment_files/figure-html/unnamed-chunk-5-1.png) 

## Fitting Models to the Data
Several models were fit to the data. 
Models were chosen by first evaluating their performance on smaller data sets.
Models with relatively high accuracy and reasonable computation times were used for further analysis with larger data sets.
The relative accuracy for various models was tested using the confusionMatrix function for model predictions on the validation data set.


```r
# Keep some statistics to compare models
eltime <- c()
acc <- list()
```

## Conditional Inference Tree Model (ctree)

```r
pt1 <- proc.time()
model_ctree <- train(classe ~ ., data=df_training,method="ctree",preProcess=c("center","scale"))
eltime["ctree"] = (proc.time()-pt1)[3]

pred_ctree <- predict(model_ctree, df_validation)
cm_ctree <- confusionMatrix(pred_ctree,df_validation$classe)
acc[["ctree"]] <- cm_ctree$overall
```

### K-Nearest Neighbor Model (knn)

```r
pt1 <- proc.time()
model_knn <- train(classe~., data=df_training,method="knn",preProcess=c("center","scale"))
eltime["knn"] = (proc.time()-pt1)[3]

pred_knn <- predict(model_knn,df_validation)
cm_knn <- confusionMatrix(pred_knn,df_validation$classe)
acc[["knn"]] <- cm_knn$overall
```

### Random Forest (rf)

```r
pt1 <- proc.time()
model_rf <- train(classe~., data=df_training,method="rf",preProcess=c("center","scale"))
eltime["rf"] = (proc.time()-pt1)[3]

pred_rf <- predict(model_rf,df_validation)
cm_rf <- confusionMatrix(pred_rf,df_validation$classe)
acc[["rf"]] <- cm_rf$overall
```


## Compare Model Results

```r
df_acc <- data.frame(acc)
df_eltime <- data.frame(elapsed_time=eltime)

df_acc
```

```
##                       ctree           knn           rf
## Accuracy       7.294111e-01  7.901151e-01 9.506887e-01
## Kappa          6.566714e-01  7.341123e-01 9.375869e-01
## AccuracyLower  7.227898e-01  7.840302e-01 9.473904e-01
## AccuracyUpper  7.359567e-01  7.961041e-01 9.538374e-01
## AccuracyNull   2.843621e-01  2.843621e-01 2.843621e-01
## AccuracyPValue 0.000000e+00  0.000000e+00 0.000000e+00
## McnemarPValue  3.403274e-33 1.390351e-199 3.911290e-56
```

```r
df_eltime
```

```
##       elapsed_time
## ctree        68.16
## knn           6.39
## rf          201.44
```

## Summarize Best Model (Random Forest)

```r
model_rf
```

```
## Random Forest 
## 
## 1963 samples
##   53 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 1963, 1963, 1963, 1963, 1963, 1963, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
##    2    0.9242194  0.9041201  0.01216673   0.01536303
##   29    0.9323433  0.9144215  0.01145709   0.01449620
##   57    0.9173731  0.8955206  0.01169827   0.01471532
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 29.
```

```r
varImp(model_rf)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 57)
## 
##                      Overall
## roll_belt             100.00
## pitch_forearm          62.88
## magnet_dumbbell_z      54.56
## magnet_dumbbell_y      53.32
## yaw_belt               52.14
## roll_forearm           42.51
## pitch_belt             36.35
## roll_dumbbell          27.32
## magnet_dumbbell_x      24.62
## accel_dumbbell_y       23.37
## magnet_belt_z          20.22
## magnet_belt_y          19.78
## accel_forearm_x        19.53
## accel_belt_z           18.34
## gyros_belt_z           18.07
## total_accel_dumbbell   17.08
## magnet_belt_x          16.48
## accel_dumbbell_x       16.35
## magnet_forearm_z       16.32
## accel_dumbbell_z       16.28
```

## Predict Test Classe Behavior

```r
pred_test <- predict(model_rf,testing_sm)
pred_test
```

```
##  [1] B A B A A E D D A A B C B A E E A B B B
## Levels: A B C D E
```

## References
Data was originally made available from 
[http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har).

