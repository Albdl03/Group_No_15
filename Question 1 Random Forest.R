install.packages("randomForest")
install.packages("dplyr")
library(randomForest)
library(dplyr)
library(ggplot2)

data <- read.csv("cleaned_bank (1).csv")

head(data)
summary(data)
str(data)

# Check for missing values
sum(is.na(data))

# Convert relevant columns to factors if they are not already
data$job <- as.factor(data$job)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)
data$loan <- as.factor(data$loan)
data$housing <- as.factor(data$housing)  # The target variable

set.seed(123)  # For reproducibility
sample_index <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[sample_index, ]
test_data <- data[-sample_index, ]

# Build the model with 'housing' as the target variable
rf_model <- randomForest(housing ~ age + job + marital + education + loan + balance, 
                         data = train_data, 
                         ntree = 500,  # Number of trees
                         importance = TRUE)  # Calculate variable importance

# Check the model's summary
print(rf_model)

# Make predictions
predictions <- predict(rf_model, test_data)

# Confusion Matrix to evaluate the model's performance
table(predictions, test_data$housing)

# Get the variable importance
importance_values <- importance(rf_model)

# Convert the importance values into a data frame for easier plotting
importance_df <- data.frame(
  Feature = rownames(importance_values),
  MeanDecreaseGini = importance_values[, "MeanDecreaseGini"],
  MeanDecreaseAccuracy = importance_values[, "MeanDecreaseAccuracy"]
)

# Load ggplot2 for a more customizable bar plot
library(ggplot2)

# Create a bar plot for Mean Decrease Gini
ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # Flip coordinates to make it horizontal
  labs(title = "Feature Importance - Mean Decrease Gini",
       x = "Features",
       y = "Mean Decrease Gini") +
  theme_minimal()

# Create a bar plot for Mean Decrease Accuracy
ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = "coral") +
  coord_flip() +  # Flip coordinates to make it horizontal
  labs(title = "Feature Importance - Mean Decrease Accuracy",
       x = "Features",
       y = "Mean Decrease Accuracy") +
  theme_minimal()