# PythonAnalysis_Commerce
Analyzing E-Commerce dataset using python because coding is cool

##  Purpose & Disclaimer
Attempting to clean/manipulate data, creating simple visualizations and trying my hand at machine learning using sklearn.

This project was primarily a self-study project for fun. Although I had no prior experience with machine learning, I explored the subject by experimenting with code and following tutorials and documentation, utilizing Anaconda Assistant for support.

## Technologies/Libraries used
   - Python
   - JupyterLab
   - Numpy
   - Pandas
   - matplotlib
   - sklearn

<img width="100" src="https://github.com/user-attachments/assets/ae3de72e-7caa-4d8d-a92d-e7df13499810">
<img width="50" src="https://github.com/user-attachments/assets/859f5744-3276-45ff-8531-212b72dfa5d3">
<img width="50" src="https://github.com/user-attachments/assets/79881fd5-3528-4839-8a36-a6ee1bd0fab1">
<img width="50" src="https://github.com/user-attachments/assets/6807e23a-67ac-4fd5-b969-fd05c41be9fc">
<img width="50" src="https://github.com/user-attachments/assets/91141cd9-eab3-4cf6-b113-39a86bba1bce">

## Screenshots & Code Snippets
![image](https://github.com/user-attachments/assets/bc2acc23-d78e-453f-a651-54d640720774)


**Visualizing the Data Distribution for all columns**
```python
# distribution
for col in data.columns :
    print('_'*40)
    print(col)
    print(data[col].value_counts())
```
**Visualizing the Data Distribution using a Kernal Density Estimation (KDE)**

![image](https://github.com/user-attachments/assets/5aa25624-03d2-4b9f-a768-932e54d34067)


**Fixing Date data type**

![image](https://github.com/user-attachments/assets/209104e6-0d38-4081-a3ac-785011301d73)


**Finding Null Values and displaying and visualizing the null values in the data using a Heat Map**

![image](https://github.com/user-attachments/assets/f1bd56b8-f3d1-4343-b923-8a9b40a56b8e)

```python
# Filling null values with the most common value inside that column
for col in data.columns:
    counter = counter +1 
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)
```

**Using a boxplot to display a visual summary of the data distribution, giving insight to the median, quartiles, and potential outliers.**

![image](https://github.com/user-attachments/assets/4e7361a0-900b-4f41-a918-f68e34cb0db5)

**Visualizing the distribution using a Pie Chart**

![image](https://github.com/user-attachments/assets/9f63a6ac-bd31-4c58-8e77-290f72fed947)


**Visualizing the number of orders for each size of clothing using a Bar Graph**

![image](https://github.com/user-attachments/assets/6de36d9f-eb3e-4897-888c-ad12c179c41e)


```python
# X = data1['Fulfilment','ship-service-level','Qty','Amount','ship-postal-code','B2B','Month'] 
cols=['Fulfilment','ship-service-level','Qty','Amount','ship-postal-code','B2B','Month']

# Preparing X,Y 
# X represents everything you're using to make a prediction
# Y represents target value, the prediction we are making
X = data1[['Fulfilment','ship-service-level','Qty','Amount','ship-postal-code','B2B','Month']]
Y = data1['Courier Status'] 

# prepairing data for testing
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=45)

# creating a model
# model = LogisticRegression()
model = KNeighborsRegressor()
# model = LinearRegression()

# training the model on the data
model.fit(X_train, y_train)

#testing the model
y_pred = model.predict(X_test)

# calculating and priting model accuracy using mean squared error formula
mse = mean_squared_error(y_test, y_pred)
# print(X_test.head(7))
print('....................')
print("Mean Squared Error (MSE):", mse)

plt.scatter(y_pred, y_test )
```
**KNeighborsRegressor Model**

![image](https://github.com/user-attachments/assets/a112b5d0-3fdd-43de-a3e1-7d9f9372b2b3)


**RandomForestClassifier Model**

![image](https://github.com/user-attachments/assets/f1018233-c390-462b-b227-fa6dd553158b)



