library(dplyr)
library(caret)
library(e1071) 
library(randomForest)
library(ggplot2)
bank_data <- read.csv("C:/Users/SAYESHA/Desktop/cleaned_bank.csv")

bank_data$job <- as.factor(bank_data$job)
bank_data$marital <- as.factor(bank_data$marital)
bank_data$education <- as.factor(bank_data$education)
bank_data$default <- as.factor(bank_data$default)
bank_data$housing <- as.factor(bank_data$housing)
bank_data$loan <- as.factor(bank_data$loan)
bank_data$contact <- as.factor(bank_data$contact)
bank_data$month <- as.factor(bank_data$month)
bank_data$poutcome <- as.factor(bank_data$poutcome)
bank_data$y <- as.factor(bank_data$y)

str(bank_data)

bank_data$poutcome <- factor(bank_data$poutcome, levels = c("unknown", "failure", "other", "success"))

set.seed(123)
index <- createDataPartition(bank_data$y, p = 0.7, list = FALSE)
train_data <- bank_data[index, ]
test_data <- bank_data[-index, ]

logistic_model <- glm(y ~ campaign + pdays + previous + poutcome, data = train_data, family = "binomial")
summary(logistic_model)

predictions <- predict(logistic_model, newdata = test_data, type = "response")
predictions <- ifelse(predictions > 0.5, "yes", "no")
predictions <- as.factor(predictions)

confusionMatrix(predictions, test_data$y)

library(broom)
coef_df <- tidy(logistic_model, conf.int = TRUE)

# Plotting coefficients
library(ggplot2)
ggplot(coef_df, aes(x = reorder(term, estimate), y = estimate)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
  labs(title = "Logistic Regression Coefficients",
       x = "Predictors",
       y = "Coefficient Estimate") +
  coord_flip() +
  theme_minimal()

#random forest 
set.seed(123)
train_data <- na.omit(train_data)
rf_model <- randomForest(y ~ campaign + pdays + previous + poutcome, data = train_data, ntree = 100)
print(rf_model)

predictions_rf <- predict(rf_model, newdata = test_data)

confusionMatrix(predictions_rf, test_data$y)

