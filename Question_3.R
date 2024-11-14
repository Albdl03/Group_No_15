#import libraries
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(xgboost)

# Reload the CSV file using semicolon as the delimiter
bank_df = read.csv("C:/Users/Alberto/Desktop/TCD/FOUNDATIONS OF BUSINESS ANALYTICS/cleaned_bank.csv", stringsAsFactors = T)

# Convert a specific factor column to character
bank_df$contact <- as.character(bank_df$contact)

# Replace NA values with "unknown"
bank_df$contact[is.na(bank_df$contact)] <- "unknown"

bank_df$contact <- as.factor(bank_df$contact)

#Check the response variable distribution
table(bank_df$y)

#From the results it is an unbalance dataset

#Analysing the impact of the communication type, day, month an duration can have on client response

final_df <- bank_df[, c("contact", "day", "month", "duration", "y")]

#divide the data into training and testing data
set.seed(123)
sample <- sample.int(n = nrow(final_df), size = floor(.7*nrow(final_df)), replace = F)
train_df <- final_df[sample, ] #we select the sample randomly
validation_df  <- final_df[-sample, ] #we select the rest of the data

#divide validation in test and validation
sample <- sample.int(n = nrow(validation_df), size = floor(.5*nrow(validation_df)), replace = F)
test_df <- validation_df[sample, ] #we select the sample randomly
validation_df  <- validation_df[-sample, ] #we select the rest of the data


#select the needed variables
x_train <- train_df[,-5]
x_val <- validation_df[,-5]
x_test <- test_df[,-5]

y_train <- train_df$y
y_val <- validation_df$y
y_test <- test_df$y


#Logistic regression model

minority_class_weight <- nrow(final_df[final_df$y == "no",]) / nrow(final_df[final_df$y == "yes",])

log_reg <- glm(y~., 
               family = binomial(link = "logit"),
               data = train_df,
               weights = ifelse(train_df$y == "yes", minority_class_weight, 1))

summary(log_reg)


# Make predictions as probabilities
pred_probs_lr <- predict(log_reg, validation_df, type = "response")

# Initialize variables to store the best threshold, highest sensitivity, and specificity
best_threshold_lr <- 0
best_sensitivity_lr <- 0
best_specificity_lr <- 0

# Define a range of thresholds to test (e.g., from 0.1 to 0.9)
thresholds <- seq(0.1, 0.9, by = 0.01)

# For loop to find the best threshold
for (threshold in thresholds) {
  # Convert probabilities to binary predictions based on the current threshold
  pred_class <- ifelse(pred_probs_lr >= threshold, "yes", "no")
  
  # Create confusion matrix
  cm <- table(Predicted = pred_class, Actual = y_val)
  
  # Calculate sensitivity and specificity
  sensitivity <- cm["yes", "yes"] / (cm["yes", "yes"] + cm["no", "yes"])
  specificity <- cm["no", "no"] / (cm["no", "no"] + cm["yes", "no"])
  
  # Update best threshold if the current sensitivity + specificity is higher
  if ((sensitivity + specificity) > (best_sensitivity_lr + best_specificity_lr)) {
    best_sensitivity_lr <- sensitivity
    best_specificity_lr <- specificity
    best_threshold_lr <- threshold
  }
}

# Print the best threshold, sensitivity, and specificity
cat("Best Threshold:", best_threshold_lr, "\n")
cat("Best Sensitivity:", best_sensitivity_lr, "\n")
cat("Best Specificity:", best_specificity_lr, "\n")

# Make predictions on the test set
prob_test_lr <- predict(log_reg, test_df, type = "response")
pred_test_lr <- ifelse(prob_test_lr >= best_threshold_lr, "yes", "no")

# Convert the predicted classes into a factor with the same levels
pred_test_lr <- factor(pred_test_lr, levels = c("no", "yes"))

# Create a confusion matrix to evaluate the performance of the model
confusion_matrix <- table(Predicted = pred_test_lr, Actual = y_test)
lr_metrics <- confusionMatrix(pred_test_lr, y_test)
lr_metrics

#ROC curve

roc_lr <- pROC::roc(y_test, 
                    prob_test_lr, 
                    plot = TRUE,
                    col = "midnightblue",
                    lwd = 3,
                    auc.polygon = T,
                    auc.polygon.col = "lightblue",
                    print.auc = T)




#Random Forest model
# Set sampsize to balance classes equally
min_class_size <- min(table(train_df$y))

#fit the model
rf_model <- randomForest(y ~ ., data = train_df, 
                         ntree = 500,
                         importance = TRUE,
                         sampsize = c(min_class_size, min_class_size))

# Get predicted probabilities instead of class labels
rf_pred_prob <- predict(rf_model, validation_df, type = "prob")[,2]  # Probability of the "Yes" class

# Initialize variables to store the best threshold, highest sensitivity, and specificity
best_threshold_rf <- 0
best_sensitivity_rf <- 0
best_specificity_rf <- 0

# For loop to find the best threshold
for (threshold in thresholds) {
  # Convert probabilities to binary predictions based on the current threshold
  pred_class <- ifelse(rf_pred_prob >= threshold, "yes", "no")
  
  # Create confusion matrix
  cm <- table(Predicted = pred_class, Actual = y_val)
  
  # Calculate sensitivity and specificity
  sensitivity <- cm["yes", "yes"] / (cm["yes", "yes"] + cm["no", "yes"])
  specificity <- cm["no", "no"] / (cm["no", "no"] + cm["yes", "no"])
  
  # Update best threshold if the current sensitivity + specificity is higher
  if ((sensitivity + specificity) > (best_sensitivity_rf + best_specificity_rf)) {
    best_sensitivity_rf <- sensitivity
    best_specificity_rf <- specificity
    best_threshold_rf <- threshold
  }
}

# Print the best threshold, sensitivity, and specificity
cat("Best Threshold:", best_threshold_rf, "\n")
cat("Best Sensitivity:", best_sensitivity_rf, "\n")
cat("Best Specificity:", best_specificity_rf, "\n")

# Make predictions on the test set
prob_test_rf <- predict(rf_model, test_df, type = "prob")[,2]
pred_test_rf <- ifelse(prob_test_rf >= best_threshold_rf, "yes", "no")

# Convert the predicted classes into a factor with the same levels
pred_test_rf <- factor(pred_test_rf, levels = c("no", "yes"))


# Create a confusion matrix to evaluate the performance of the model
confusion_matrix <- table(Predicted = pred_test_rf, Actual = y_test)
rf_metrics <- confusionMatrix(pred_test_rf, y_test)
rf_metrics

#Variable importance
varImpPlot(rf_model, main = "Variable Importance Plot")

#ROC curve

roc_rf <- pROC::roc(y_test, 
                    prob_test_rf, 
                    plot = TRUE,
                    col = "midnightblue",
                    lwd = 3,
                    auc.polygon = T,
                    auc.polygon.col = "lightblue",
                    print.auc = T)



#Naive Bayes model
# Fit the model
nb_model <- naiveBayes(y ~ ., data = train_df)

# Get predicted probabilities instead of class labels
nb_pred_prob <- predict(nb_model, validation_df, type = "raw")[,2]

# Initialize variables to store the best threshold, highest sensitivity, and specificity

best_threshold_nb <- 0
best_sensitivity_nb <- 0
best_specificity_nb <- 0

# For loop to find the best threshold

for (threshold in thresholds) {
  # Convert probabilities to binary predictions based on the current threshold
  pred_class <- ifelse(nb_pred_prob >= threshold, "yes", "no")
  
  # Create confusion matrix
  cm <- table(Predicted = pred_class, Actual = y_val)
  
  # Calculate sensitivity and specificity
  sensitivity <- cm["yes", "yes"] / (cm["yes", "yes"] + cm["no", "yes"])
  specificity <- cm["no", "no"] / (cm["no", "no"] + cm["yes", "no"])
  
  # Update best threshold if the current sensitivity + specificity is higher
  if ((sensitivity + specificity) > (best_sensitivity_nb + best_specificity_nb)) {
    best_sensitivity_nb <- sensitivity
    best_specificity_nb <- specificity
    best_threshold_nb <- threshold
  }
}

# Print the best threshold, sensitivity, and specificity
cat("Best Threshold:", best_threshold_nb, "\n")
cat("Best Sensitivity:", best_sensitivity_nb, "\n")
cat("Best Specificity:", best_specificity_nb, "\n")

# Make predictions on the test set
prob_test_nb <- predict(nb_model, test_df, type = "raw")[,2]
pred_test_nb <- ifelse(prob_test_nb >= best_threshold_nb, "yes", "no")

# Convert the predicted classes into a factor with the same levels
pred_test_nb <- factor(pred_test_nb, levels = c("no", "yes"))

# Create a confusion matrix to evaluate the performance of the model
confusion_matrix <- table(Predicted = pred_test_nb, Actual = y_test)
nb_metrics <- confusionMatrix(pred_test_nb, y_test)
nb_metrics

#ROC curve

roc_nb <- pROC::roc(y_test, 
                    prob_test_nb, 
                    plot = TRUE,
                    col = "midnightblue",
                    lwd = 3,
                    auc.polygon = T,
                    auc.polygon.col = "lightblue",
                    print.auc = T)


#XGBoost model
# Convert the response variable to a numeric binary variable
train_df$y_num <- ifelse(train_df$y == "yes", 1, 0)
validation_df$y_num <- ifelse(validation_df$y == "yes", 1, 0)
test_df$y_num <- ifelse(test_df$y == "yes", 1, 0)

#define predictor and response variables
train_x = data.matrix(x_train)
val_x = data.matrix(x_val)
test_x = data.matrix(x_test)

set.seed(3)

#Note that the xgboost package also uses matrix data, so we’ll use the data.matrix() 
xgb_train = xgb.DMatrix(data = train_x, label = train_df$y_num)
xgb_val = xgb.DMatrix(data = val_x, label = validation_df$y_num)

# Fit the XGBoost model
xgb_model <- xgboost(data = xgb_train, 
                     max.depth = 3, 
                     nrounds = 200,
                     verbose = 0, 
                     objective = "binary:logistic")

# Get predicted probabilities instead of class labels
xgb_pred_prob <- predict(xgb_model, xgb_val)

# Initialize variables to store the best threshold, highest sensitivity, and specificity
best_threshold_xgb <- 0
best_sensitivity_xgb <- 0
best_specificity_xgb <- 0

# For loop to find the best threshold
for (threshold in thresholds) {
  # Convert probabilities to binary predictions based on the current threshold
  pred_class <- ifelse(xgb_pred_prob >= threshold, "yes", "no")
  
  # Create confusion matrix
  cm <- table(Predicted = pred_class, Actual = y_val)
  
  # Calculate sensitivity and specificity
  sensitivity <- cm["yes", "yes"] / (cm["yes", "yes"] + cm["no", "yes"])
  specificity <- cm["no", "no"] / (cm["no", "no"] + cm["yes", "no"])
  
  # Update best threshold if the current sensitivity + specificity is higher
  if ((sensitivity + specificity) > (best_sensitivity_xgb + best_specificity_xgb)) {
    best_sensitivity_xgb <- sensitivity
    best_specificity_xgb <- specificity
    best_threshold_xgb <- threshold
  }
}

# Print the best threshold, sensitivity, and specificity
cat("Best Threshold:", best_threshold_xgb, "\n")
cat("Best Sensitivity:", best_sensitivity_xgb, "\n")
cat("Best Specificity:", best_specificity_xgb, "\n")

# Make predictions on the test set
prob_test_xgb <- predict(xgb_model, xgb.DMatrix(data = test_x))
pred_test_xgb <- ifelse(prob_test_xgb >= best_threshold_xgb, "yes", "no")

# Convert the predicted classes into a factor with the same levels
pred_test_xgb <- factor(pred_test_xgb, levels = c("no", "yes"))

# Create a confusion matrix to evaluate the performance of the model
confusion_matrix <- table(Predicted = pred_test_xgb, Actual = y_test)
xgb_metrics <- confusionMatrix(pred_test_xgb, y_test)
xgb_metrics

#ROC curve

roc_xgb <- pROC::roc(y_test, 
                     prob_test_xgb, 
                     plot = TRUE,
                     col = "midnightblue",
                     lwd = 3,
                     auc.polygon = T,
                     auc.polygon.col = "lightblue",
                     print.auc = T)

