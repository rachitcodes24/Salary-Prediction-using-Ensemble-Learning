import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load and explore the dataset
df = pd.read_csv("employee_data_with_locations.csv")
# For real projects, always check the data
#print(df.head())
#print(df.info())

# Drop the ID (not useful for prediction)
df = df.drop(columns=['ID'])

# Choose input variables and target
feature_cols = ['Gender', 'Experience (Years)', 'Position', 'Location']
X = df[feature_cols]
y = df['Salary']

# Convert categorical data with one-hot encoding
cat_features = ['Gender', 'Position', 'Location']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(X[cat_features])
X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_features), index=X.index)

# Merge numerical and categorical features
X_num = X.drop(columns=cat_features)
X_ready = pd.concat([X_num, X_cat_df], axis=1)

# Split into train and test for unbiased evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_ready, y, test_size=0.2, random_state=7
)

# Build and fit several ensemble models
rf_model = RandomForestRegressor(random_state=7)
gb_model = GradientBoostingRegressor(random_state=7)
vote_model = VotingRegressor([('rf', rf_model), ('gb', gb_model)])

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
vote_model.fit(X_train, y_train)

# Helper function for model evaluation
def show_metrics(model, xx, yy):
    pred = model.predict(xx)
    print("MAE: {:.1f}".format(mean_absolute_error(yy, pred)))
    print("MSE: {:.1f}".format(mean_squared_error(yy, pred)))
    print("R^2: {:.3f}".format(r2_score(yy, pred)))

print("Random Forest performance:")
show_metrics(rf_model, X_test, y_test)
print("\nGradient Boosting performance:")
show_metrics(gb_model, X_test, y_test)
print("\nVoting Regressor (ensemble) performance:")
show_metrics(vote_model, X_test, y_test)

# Example prediction - customize as needed
test_input = pd.DataFrame([{
    'Gender': 'F',
    'Experience (Years)': 7,
    'Position': 'Systems Analyst',
    'Location': 'London'
}])

test_cat = encoder.transform(test_input[cat_features])
test_cat_df = pd.DataFrame(test_cat, columns=encoder.get_feature_names_out(cat_features))
test_ready = pd.concat([test_input.drop(columns=cat_features), test_cat_df], axis=1)
estimated_salary = vote_model.predict(test_ready)[0]
print("\nSample prediction (should be sanity-checked by recruiter):")
print("Estimated salary: ${:,.0f}".format(estimated_salary))

# Save model and encoder for future use
joblib.dump(vote_model, "salary_predictor.pkl")
joblib.dump(encoder, "feature_encoder.pkl")
