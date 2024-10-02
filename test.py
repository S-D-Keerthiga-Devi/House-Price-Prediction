import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['price'] = california.target
df.drop(columns=['AveBedrms', 'Population'],axis=1,inplace=True)

# Dependent and Independent features
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# standardization of the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#model training
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

# cross validation
from sklearn.model_selection import cross_val_score
validation_score = cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=3)

# prediction with the test data
y_pred = regression.predict(X_test)

# Residuals
residuals = y_test - y_pred

# Performance metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)

# OLS linear regression
import statsmodels.api as sm
model = sm.OLS(y_train,X_train).fit()

# Page setup
st.set_page_config(page_title="California Housing Price Prediction", page_icon=":house:", layout="wide")

# Title
st.write("""
    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; text-align:left;'>
        <h1 style='text-align: center; color: #FF6347;'>House Price Prediction</h1>
    </div>
    """, unsafe_allow_html=True)

# Description of the dataset
st.header("Description")
with st.expander("Click Below to view dataset description"):
    st.write(california.DESCR)

# Prediction of new data
st.header("Prediction")
@st.dialog("Predicted House Price")
def add(data):
    st.write("### Predicted House Price: ")
    new_data = pd.DataFrame(data,index=[0])
    new_data = new_data.astype(np.float64)
    new_data_scaled = scaler.transform(new_data)
    new_pred = regression.predict(new_data_scaled)
    for i,pred in enumerate(new_pred):
        st.write(f"$ {pred:.2f} hundred thousand dollars")
        if st.button("OK"):
            st.session_state.add = {"predict":pred}
            st.rerun()

data ={}
if "add" not in st.session_state:
    st.write("### Enter values to predict the house price:")
    for i,feature in enumerate(california.feature_names):
        if feature != 'AveBedrms' and feature != 'Population':
            data[feature] = st.number_input(feature,value=0.0)
    if st.button("Submit"):
        add(data)
else:
    f"The predicted house price is: $ {st.session_state.add['predict']:.2f} hundred thousand dollars"
st.divider()

# Scatter plot of the Actual vs Predicted House Prices
chart_data = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
st.header("Scatter plot of Actual vs Predicted House Prices")
st.scatter_chart(chart_data)

st.divider()

st.header("Linear Regression Model Analysis")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Actual vs Predicted", "coefficients", "OLS Summary", "Residuals", "Prediction vs Residuals", "R-squared", "Adjusted R-squared"])

with tab1:
    st.header("Scatter Plot of Actual and Predicted House Prices")
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_test, y_test, color='blue', label='Actual Prices',alpha=0.5)  
    ax.scatter(y_test, y_pred, color='red', label='Predicted prices',alpha=0.5)
    ax.set_xlabel('Actual House Prices')
    ax.set_ylabel('Predicted House Prices')
    ax.set_title('Actual vs Predicted House Prices')
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.header("Linear Regression Coefficients and Intercept")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Coefficients")
        coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": regression.coef_})
        st.dataframe(coef_df)

    with col2:
        st.subheader("Intercept")
        st.write(f"### {regression.intercept_:.4f}")
with tab3:
    st.header("OLS (Ordinary Least Squares) Summary")
    st.write(model.summary())
    with st.popover("Note"):
        st.markdown("The Linear Regression model and the OLS method produce the same coefficients.")

with tab4:
    st.header("Residual Error Distribution")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.kdeplot(residuals,ax=ax,fill=True,color='orange')
    st.pyplot(fig)

with tab5:
    st.header("Residual Analysis: Predicted Values vs Residuals")
    fig, ax = plt.subplots(figsize=(8,6)) 
    ax.scatter(y_pred, residuals, color='red', label='Residuals')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Predicted Values vs Residuals')
    ax.legend()
    st.pyplot(fig)

with tab6:
    st.header("R-squared formula")
    st.latex(r'R^2 = 1 - \frac{SSR}{SST}')
    st.markdown('''### where:

        R2: The Coefficient of Determination\n
        SSR: The Sum of Squared Residuals\n
        ST: The Total Sum of Squares''')
    st.write(f"### After applying the formula, we arrive at a result of **{score:.4f}**, indicating the model's performance.")

with tab7:
    st.header("Adjusted R-squared formula")
    st.latex(r'Adjusted \ R^2 = 1 - \left(\frac{(1-R^2)(n-1)}{(n-k-1)} \right)')
    st.markdown('''### where:

        R2: The Coefficient of Determination\n
        n: The Number of Observations\n
        k: The Number of Predictor Variables''')
    adjusted_r2 = 1 - (1 - score) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    st.write(f"### After applying the formula, we arrive at an Adjusted R-squared value of **{adjusted_r2:.4f}**, which indicates the model's performance.")


# Exploratory Data Analysis (EDA)
st.header("Exploratory Data Analysis (EDA) of California Housing Dataset")
t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Sample data","Statstical Summary","Data Types","Correlation","Correlation Heatmap","Median Income vs House Price","Population vs House Price"])
with t1:
    st.subheader("The First Ten Rows of the Dataset")
    st.write(df.head(10))
with t2:
    st.subheader("Descriptive Statstical Summary")
    st.write(df.describe())
with t3:
    st.subheader("Data Type Information")
    st.write(df.dtypes)
with t4:
    st.subheader("Correaltion Among variables")
    st.write(df.corr())
with t5:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(),annot=True,fmt='.2f',cmap="coolwarm",cbar=True)
    plt.title('Correlation Heatmap of California Housing Dataset')
    st.pyplot(fig)
with t6:
    st.subheader("Plotting Median Income vs House Price")
    fig, ax = plt.subplots(figsize=(8,6)) 
    ax.scatter(x='MedInc', y='price', data=df)
    ax.set_xlabel('Median Income')
    ax.set_ylabel('House Price')
    ax.set_title('Median Income vs House Price')
    ax.legend()
    st.pyplot(fig)

with t7:
    st.subheader("Plotting House Age vs House Price")
    fig, ax = plt.subplots(figsize=(8,6)) 
    ax.scatter(x='HouseAge', y='price', data=df)
    ax.set_xlabel('House Age')
    ax.set_ylabel('House Price')
    ax.set_title('House Age vs House Price')
    ax.legend()
    st.pyplot(fig)