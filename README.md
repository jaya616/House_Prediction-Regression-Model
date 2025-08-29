# House_Prediction-Regression-Model
The House Price Prediction Model is a web-based machine learning project that estimates house prices using regression analysis. The project allows users to upload a housing dataset (CSV format) containing features of houses along with their sale prices.

Once uploaded, the system automatically preprocesses the data by filling missing values, converting categorical variables into numeric form, and normalizing features.

The core of the project is a linear regression model, implemented using gradient descent in JavaScript. The dataset is split into training and testing sets, and the model is trained on the training set. 
After training, the system evaluates its performance with key metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R² score. These results are presented in a clean dashboard, along with a scatter plot comparing actual and predicted house prices.

Finally, the model enables users to make predictions for new houses. A dynamic form is generated where users can input important house features, and the system predicts the likely selling price.

This project demonstrates the integration of machine learning and web technologies, providing an interactive, serverless solution where all training and predictions are performed directly in the browser.
