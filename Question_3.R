#import libraries
library(caret)
library(pROC)
library(randomForest)

#QUESTION 3
#Analysing the impact of the communication type, day, month an duration can have on client response


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
