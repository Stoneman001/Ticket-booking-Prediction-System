import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Sample data
data = {
    'day_of_week': [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2],  # Repeated for each event type
    'event_type': [1]*7 + [2]*7 + [3]*7 + [4]*7 + [5]*2,  # Event types from 1 to 5
    'demand': [50, 60, 30, 40, 70, 20, 25, 30, 45, 35, 40, 55, 65, 70, 45, 55, 35, 45, 50, 60, 25, 30, 40, 45, 50, 55, 60, 65, 70, 75]  # Example demand values for each day/event type
}

df = pd.DataFrame(data)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['day_of_week', 'event_type'])

X = df.drop('demand', axis=1)
y = df['demand']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'ticket_demand_model.pkl')
