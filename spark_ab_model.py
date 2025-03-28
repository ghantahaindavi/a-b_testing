from pyspark.sql import SparkSession
import pandas as pd
from econml.dr import DRLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

spark = SparkSession.builder.appName("EcomABTestEnhanced").getOrCreate()

def process(path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    pandas_df = df.toPandas()

    # Encode categorical variables
    pandas_df = pd.get_dummies(pandas_df, columns=["location", "device", "gender"], drop_first=True)

    features = ['time_spent', 'age', 'days_since_last_visit', 'pages_viewed'] +                [col for col in pandas_df.columns if col.startswith("location_") or col.startswith("device_") or col.startswith("gender_")]
    
    X = pandas_df[features]
    T = pandas_df['group'].map({'control': 0, 'treatment': 1})
    Y = pandas_df['converted']

    model_y = RandomForestRegressor()
    model_t = LogisticRegression()
    learner = DRLearner(model_regression=model_y, model_propensity=model_t)
    learner.fit(Y, T, X=X)

    pandas_df['uplift'] = learner.effect(X)
    return pandas_df
