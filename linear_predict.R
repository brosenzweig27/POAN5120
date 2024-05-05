rm(list = ls())
setwd("/Users/bennyrose/Desktop/Columbia/Spring 2024/Predictive Modeling/Final Project")

all <- read_csv("cleaned.csv")

# Split the data into training and testing sets (e.g., 70% train, 30% test)
set.seed(2342)
train_index <- sample(seq_len(nrow(all)), 0.5 * nrow(all))
train_data <- all[train_index, ]
test_data <- all[-train_index, ]

# Create a linear regression model
model <- lm(vote_g2022 ~ . - vote_p2022, data = train_data)

# Predict 'vote_g2024' on the testing set
predicted_values <- predict(model, newdata = test_data)

# Calculate prediction accuracy
actual_values <- test_data$vote_g2022
correct_predictions <- sum(round(predicted_values) == actual_values)
total_predictions <- length(actual_values)
accuracy <- correct_predictions / total_predictions * 100

# Print the prediction accuracy
print(paste("Prediction Accuracy:", accuracy, "%"))