# Load necessary libraries
library(readr)
library(dplyr)
library(caret) # For data splitting

banks <- read.csv("/Users/miguelmunozgonzalez/Downloads/cleaned_bank (1).csv")


# Inspect the data
View(banks)

# Convert categorical variables to factors if needed
banks$job <- as.factor(banks$job)
banks$marital <- as.factor(banks$marital)
banks$education <- as.factor(banks$education)
banks$loan <- as.factor(banks$loan)
banks$housing <- as.factor(banks$housing)  # This is your outcome variable

# Set a seed for reproducibility
set.seed(123)

# Split the data: 70% for training and 30% for validation
train_index <- createDataPartition(banks$housing, p = 0.7, list = FALSE)
train_data <- banks[train_index, ]
validation_data <- banks[-train_index, ]

# Fit the logistic regression model on the training data
model <- glm(housing ~ age + job + marital + education + loan + balance,
             data = train_data, 
             family = binomial)
# Predict on the validation set
validation_predictions <- predict(model, validation_data, type = "response")

# Convert probabilities to binary outcomes (using a threshold of 0.5)
validation_data$predicted_housing <- ifelse(validation_predictions > 0.5, "yes", "no")

# Convert to factor to match the actual data format
validation_data$predicted_housing <- as.factor(validation_data$predicted_housing)

# Confusion matrix
confusionMatrix(validation_data$predicted_housing, validation_data$housing)

#Check for the best Threshold

# Load necessary libraries
library(caret)
library(pROC)  # For ROC and optimal threshold calculation

# Calculate the ROC curve based on validation predictions and actual outcomes
roc_obj <- roc(validation_data$housing, validation_predictions)

# Plot the ROC curve (optional)
plot(roc_obj, main="ROC Curve", col="blue", lwd=2)

# Find the optimal threshold based on Youden's J statistic
optimal_threshold <- coords(roc_obj, "best", ret="threshold", transpose = TRUE)
cat("Optimal threshold:", optimal_threshold, "\n")

# Apply the optimal threshold to convert probabilities to binary outcomes
validation_data$optimal_predicted_housing <- ifelse(validation_predictions > optimal_threshold, "yes", "no")

# Convert to factor for comparison with actual values
validation_data$optimal_predicted_housing <- as.factor(validation_data$optimal_predicted_housing)

# Confusion matrix for the new threshold
confusionMatrix(validation_data$optimal_predicted_housing, validation_data$housing)

# Check for unbalanced data
table(banks$housing)

# Calculate imbalance
table_housing <- table(banks$housing)
imbalance_ratio <- max(table_housing) / min(table_housing)
cat("Class imbalance ratio:", imbalance_ratio, "\n")

# plot imbalance
library(ggplot2)
ggplot(banks, aes(x = housing)) + 
  geom_bar() +
  labs(title = "Distribution of Housing Variable")

# Show model coefficients
summary(model)
