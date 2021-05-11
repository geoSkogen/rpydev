dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$VarD, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)
# 
# training_set = scale(training_set)
# test_set = scale(test_set)

# lin_reg = lm(formula = VarD ~ ., data = dataset)
# 
# dataset$VarI2 = dataset$VarI^2
# dataset$varI3 = dataset$VarI^3
# 
# poly_reg = lm(formula = Salary ~ ., data = dataset)

#support vector regression
#library(e1071)
#regressor = svm(formula = Salary ~ .,
#                data = dataset,
#                type = 'eps-regression',
#                kernel = 'radial')

#y_pred = predict(regressor, data.frame(Level = 6.5))


#decision tree regression
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset, 
                  control = rpart.control(minsplit = 1))

#scatterplot w/ regression line
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
# swao x_grid for x to increase resolution
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata=data.frame(Level = x_grid))),color='blue') +
  ggtitle('blah') + xlab('doo') + ylab('bop')

