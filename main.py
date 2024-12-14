import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("taxi_trip_pricing.csv")

num_duplicates = df.duplicated().sum()
print(f"Count of duplicates: {num_duplicates}")

df['Trip_Distance_km'].fillna(df['Trip_Distance_km'].median(), inplace=True)
df['Passenger_Count'].fillna(df['Passenger_Count'].median(), inplace=True)
df['Trip_Price'].fillna(df['Trip_Price'].median(), inplace=True)

df['Time_of_Day'] = df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0])
df['Day_of_Week'] = df['Day_of_Week'].fillna(df['Day_of_Week'].mode()[0])
df['Traffic_Conditions'] = df['Traffic_Conditions'].fillna(df['Traffic_Conditions'].mode()[0])
df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])

label_encoder = LabelEncoder()

df["Time_of_Day"] = label_encoder.fit_transform(df["Time_of_Day"])
df["Day_of_Week"] = label_encoder.fit_transform(df["Day_of_Week"])
df["Traffic_Conditions"] = label_encoder.fit_transform(df["Traffic_Conditions"])
df["Weather"] = label_encoder.fit_transform(df["Weather"])

df = df.drop(labels= ["Base_Fare", "Per_Km_Rate", "Per_Minute_Rate", "Trip_Duration_Minutes"],axis=1)

X = df.drop(['Trip_Price'], axis = 1)
y = df['Trip_Price']

feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state=100) 

regression_model = LinearRegression()

model = regression_model.fit(X_train, y_train)

print(model.feature_names_in_)

inputs = []
# input_labels = {'Trip_Distance_km' : [0,1,2,3], 'Time_of_Day', 'Day_of_Week','Passenger_Count', 'Traffic_Conditions', 'Weather'}

trip_distance = int(input("enter the trip distance: "))
time_of_the_day = int(input("""When are you travelling 0 : Morning, 1 : Afternoon,2 : Evining, 3: Night"""))

day_of_week = int(input("""Which day of the week are you travelling 0: Weekday, 1: Weekend"""))

passenger_count = int(input("Enter the number of passengers travelling: "))


traffic_conditions = int(input("""What is the current traffic condition 0 : Low, 1 : High, 2 : Medium : """))


weather = int(input("""What is the weather condition: 0 : Clear, 1 : Rain, 2 : Snow: """))

y_hat = model.predict([[trip_distance, time_of_the_day, day_of_week, passenger_count, traffic_conditions, weather]])

print(f"The charge for this trip would be {y_hat[0]} rupees/-")