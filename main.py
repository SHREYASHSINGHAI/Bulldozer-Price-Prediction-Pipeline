################ IMPORTS #########################
print("Importing required models...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_log_error
from sklearn.model_selection import RandomizedSearchCV 
print("Models imported.")

### Importing Dataset
df=pd.read_csv(r'bluebook-for-bulldozers\TrainAndValid.csv',low_memory=False,parse_dates=['saledate'])
df_copy=df.copy()

#Exploring dataset
print(df_copy.info())
print("\n Null values in the dataset : ")
print(df_copy.isna().sum())
print(" ")


################ DATA CLEANING AND PREPROCESSING ######################
print("Starting Data Cleaning and Preprocessing...")
### sort dataframe by sale date 
df_copy.sort_values(by=["saledate"],inplace=True,ascending=True)
df_copy["state"]=df_copy["state"].replace('Washington DC','Washington')

### Visualization
df_copy['SalePrice'].plot.hist(color='brown')
plt.title("Sale Price Frequency")
plt.xlabel('Price')
plt.show()

fig,ax=plt.subplots()
ax.scatter(df_copy['saledate'][:1000],df_copy['SalePrice'][:1000])
plt.title('Year Wise Sales')
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()

############### FEATURE ENGINEERING ###################
print("Starting Feature Engineering...")
def FeatureEngineering():
    df_copy['sale_year']=df_copy.saledate.dt.year
    df_copy['sale_month'] =df_copy.saledate.dt.month
    df_copy['sale_day_of_week'] =df_copy.saledate.dt.dayofweek
    df_copy['sale_day_of_year'] =df_copy.saledate.dt.dayofyear
    df_copy.drop("saledate",axis=1,inplace=True)

    print("\n Columns : ")
    print(df_copy.columns)
    print(df_copy.state.value_counts())

    print("\n The string type columns are : ")
    for label,content in df_copy.items():
        if not pd.api.types.is_numeric_dtype(content):
            print(label)
    print("Converting string type columns to category....")
    for label,content in df_copy.items():
        if pd.api.types.is_string_dtype(content):
            df_copy[label] = content.astype('category').cat.as_ordered()
    print("\n Converted column values form string to category : ")
    print(df_copy.state.cat.categories)
    print("Feature engineering completed.")

FeatureEngineering()

######## HANDLING MISSING DATA ################
##### Handling Numerical Data ####
print("Handling numerical missing data...")
print(df_copy.isna().sum())
for label,content in df_copy.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            #Added a binary column to know whether column was null
            df_copy[label+"_is_missing"]=pd.isnull(content)
            #Filling missing values with median
            df_copy[label] = content.fillna(content.median())
print("\n Missing numerical values handeled sucessfully.")

for label,content in df_copy.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)

#check to see how many examples were missing
print(df_copy.auctioneerID_is_missing.value_counts())

##### Handling Categorical Data #######
print("Handling catagorical missing values...")
print("Categorical columns with missing values are: ")
for label,content in df_copy.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)

print("\n Codes for categorical column state are: ")
print(pd.Categorical(df_copy["state"]).codes)
# Converting catagorical colunmns to numerical  and filling missing data
for label,content in df_copy.items():
    if not pd.api.types.is_numeric_dtype(content):
        #adding extra binary column to recognise which values were filled
        df_copy[label+"_is_missing"]=pd.isnull(content)
        df_copy[label]=pd.Categorical(content).codes+1 #as it assign -1 to missing value, we are making it 0
print("Missing catagorical values handeled succesfully")


############### SPLITTING DATA ########################

df_val = df_copy[df_copy.sale_year==2012]
df_train = df_copy[df_copy.sale_year!=2012]

x_train,y_train=df_train.drop("SalePrice",axis=1),df_train.SalePrice
x_valid,y_valid=df_val.drop("SalePrice",axis=1),df_val.SalePrice

################## EVALUATION FUNCTION ######################

def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def show_scores(model):
    train_preds=model.predict(x_train)
    val_preds=model.predict(x_valid)
    scores = {"Training MAE" : mean_absolute_error(y_train,train_preds),
              "Valid MAE" : mean_absolute_error(y_valid,val_preds),
              "Training RMSL" : rmsle(y_train,train_preds),
              "Valid RMSL" : rmsle(y_valid,val_preds)}
    return scores

############## SOME EXPERIMENTAL MODELS ###########################

# Randomized search cv with Random Forest
rf_grid={"n_estimators": np.arange(10,100,10),
         "max_depth":[None,3,5,10],
         "min_samples_split": np.arange(2,20,2),
         "min_samples_leaf": np.arange(1,20,2),
        "max_features":[0.5,1,'sqrt'] }

print("Applying randomised search cv with Random Forest Regressor... ")
rs_model=RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,random_state=42),param_distributions=rf_grid,n_iter=5,cv=5,verbose=True)
rs_model.fit(x_train,y_train)
print("Randomized search cv scores : ")
print(show_scores(rs_model))

# RandomForest with ideal params
print("Applying Random Forest Regressor with ideal params...")
ideal_model=RandomForestRegressor(n_estimators=40,
                                  min_samples_leaf=1,
                                  min_samples_split=14,
                                  max_features=0.5,
                                  n_jobs=-1,
                                  max_samples=None,
                                  random_state=42)
ideal_model.fit(x_train,y_train)
print("Ideal model scores: ")
print(show_scores(ideal_model))

################## preprocessing test data #####################
df_test=pd.read_csv(r"D:\Project 2\bluebook-for-bulldozers\Test.csv",low_memory=False,parse_dates=["saledate"])
def preprocess_data(dataframe):
    dataframe['sale_year']=dataframe.saledate.dt.year
    dataframe['sale_month'] =dataframe.saledate.dt.month
    dataframe['sale_day_of_week'] =dataframe.saledate.dt.dayofweek
    dataframe['sale_day_of_year'] =dataframe.saledate.dt.dayofyear
    dataframe.drop("saledate",axis=1,inplace=True)

    ### Handling numeric missing values
    print("Handling numerical missing data for test dataset...")
    for label,content in dataframe.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
            #Added a binary column to know whether column was null
             dataframe[label+"_is_missing"]=pd.isnull(content)
            #Filling missing values with median
             dataframe[label] = content.fillna(content.median())

    ### Handling catagorical data
        print("Handling catagorical missing data... ")
        if not pd.api.types.is_numeric_dtype(content):
            dataframe[label+"_is_missing"] = pd.isnull(content)
            dataframe[label]=pd.Categorical(content).codes+1
        print("Catagorical missing values for test data handled successfully")
    return dataframe

df_test["auctioneerID_is_missing"]=False
print("Preprocessing test data...")
preprocess_data(df_test)
print("Preprocessed test data successfully.")

# Ensure test data has EXACT same columns and order as training data
df_test = df_test[x_train.columns]

print("Predicting test values....")
test_preds=ideal_model.predict(df_test)
print(test_preds)
print("Prediction Successful")

df_preds=pd.DataFrame()
df_preds["SalesID"]=df_test["SalesID"]
df_preds["SalesPrice"]=test_preds
df_preds.to_csv("bluebook-for-bulldozers/Test-Predictions.csv",index=False)



############# FEATURE IMPORTANCE #################
def plot_features(columns,importances,n=20):
    df=(pd.DataFrame({"Features":columns,
                     "Feature_Importances":importances}).sort_values("Feature_Importances",ascending=False).reset_index(drop=True))
    fig,ax=plt.subplots()
    ax.barh(df["Features"][:n],df["Feature_Importances"][:,n])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature Importance")
    ax.invert_yaxis()
plot_features(x_train.columns,ideal_model.feature_importances_)
plt.show()