#Load the data 
data <- read.csv("~/TCD Study/Foundation/cleaned_bank.csv")#Load library
library(caret)
library(rpart)
library(rpart.plot)
library(tidyverse)

#DATA PREPARATION 
#Convert Ctaegorical variables to factors

data$job <- as.factor(data$job)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)
data$default <- as.factor(data$default)
data$housing <- as.factor(data$housing)
data$loan <- as.factor(data$loan)
data$contact <- as.factor(data$contact)
data$poutcome <- as.factor(data$poutcome)
data$y <- as.factor(data$y)

#Split the data into training and test set
set.seed(123)
trainIndex <- createDataPartition(data$y, p = .8, 
                                  list = FALSE, 
                                  times = 1)
data_train <- data[ trainIndex,]
data_test  <- data[-trainIndex,]


# Build classification tree model
tree_model <- rpart(y ~ ., data = data_train, method = "class")
rpart.plot(tree_model)

# Predict and evaluate on test data
tree_preds <- predict(tree_model, newdata = data_test, type = "class")
confusionMatrix(tree_preds, data_test$y)


#Build loiistic regression model
logistic_model <- glm(y ~ ., data = data_train, family = "binomial")
summary(logistic_model)

# Predict and evaluate on test data
logistic_preds <- predict(logistic_model, newdata = data_test, type = "response")
logistic_preds <- ifelse(logistic_preds > 0.5, "yes", "no")
logistic_preds <- factor(logistic_preds, levels = c("no", "yes"))
confusionMatrix(logistic_preds, data_test$y)

# Load necessary libraries
library(caret)
library(pROC)

# Predict probabilities for the positive class (e.g., "yes") for the logistic model
logistic_probs <- predict(logistic_model, newdata = data_test, type = "response")

# Ensure that the actual values are factors with levels "no" and "yes"
data_test$y <- factor(data_test$y, levels = c("no", "yes"))

# Generate the ROC curve for the logistic model using probabilities
log_roc <- roc(data_test$y, logistic_probs)
plot(log_roc, main = "ROC Curve for Logistic Regression Model")
print(auc(log_roc))  # Print the AUC for logistic regression

# Comparison with a decision tree model, assuming tree_preds are available:
# Convert tree_preds to numeric if they're not probabilities

tree_probs <- as.numeric(predict(tree_model, newdata = data_test, type = "prob")[, "yes"])

# # Generate the ROC curve for the decision tree model

tree_roc <- roc(data_test$y, tree_probs)
plot(tree_roc, col = "yellow", add = TRUE)
print(auc(tree_roc))
