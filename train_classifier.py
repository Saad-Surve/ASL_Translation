import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Identify the maximum length of any sublist in data
max_length = max(len(sublist) if sublist else 0 for sublist in data)

# Pad each sublist to the maximum length, creating a 2D array
data_padded = [sublist + [np.nan] * (max_length - len(sublist)) if sublist else [np.nan] * max_length for sublist in data]

# Convert the list of lists to a 2D NumPy array
data_flat = np.array(data_padded)

# Impute missing values (NaN) using mean strategy
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_flat)

# Filter out rows with all NaN values
# data_imputed = data_imputed[~np.all(np.isnan(data_imputed), axis=1)]

x_train, x_test, y_train, y_test = train_test_split(data_imputed, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print(score)
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
