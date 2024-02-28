import pandas as pd

from sklearn.model_selection import train_test_split

from flask import Flask, request

import sweetviz as sv

import joblib

from ydata_profiling import ProfileReport


df = pd.read_csv('https://archive.ics.uci.edu/static/public/109/data.csv')

# Add `type` column to `red` with price one

df.rename(columns = {'class':'Type'}, inplace = True)

typeEncoder = {
    1 : 'Barolo',
    2: 'Grignolino',
    3: 'Barbera'
}

x = df.drop(columns='Type')
y = df['Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# model = keras.models.load_model('models/wine_predict_model.keras')

model=joblib.load("model.pkl")
model.fit(x_train, y_train)

profile = ProfileReport(df)

my_report = sv.analyze(df)
my_report.show_html()

profile.to_file('report.html')
def predict_wine_type_from_data(
        alcohol,
        malicacid,
        ash,
        alcalinity_of_ash,
        magnesium,
        total_phenols,
        flavanoids,
        non_flavanoid_phenols,
        proanthocyanins,
        color_intensity,
        hue,
        diluted_wines_quantity,
        proline
    ):
    # new_data = [[7.5,0.69,0.00,1.9,0.079,23,60,0.9962,3.18,9.3,5]]
    new_data = [[ alcohol,
        malicacid,
        ash,
        alcalinity_of_ash,
        magnesium,
        total_phenols,
        flavanoids,
        non_flavanoid_phenols,
        proanthocyanins,
        color_intensity,
        hue,
        diluted_wines_quantity,
        proline]]

    print(type(new_data))
    predictions = model.predict(new_data)
    
    return round(predictions[0])

app = Flask(__name__)

@app.route("/", methods=["POST"])
def get_wine_type():
    body = request.get_json()
    prediction = predict_wine_type_from_data(body['alcohol'], body['malicacid'], body['ash'], body['alcalinity_of_ash'], body['magnesium'], body['total_phenols'], body['flavanoids'], body['non_flavanoid_phenols'], body['proanthocyanins'], body['color_intensity'], body['hue'],body['diluted_wines_quantity'], body['proline'])
    wine_name = 'Could not be determined'
    
    if(prediction in typeEncoder.keys()):
        wine_name = typeEncoder[prediction]

    return {
        'type': "%d" % prediction,
        'wine_name' : "%s" % wine_name
    }

if __name__ == '__main__':
    print(predict_wine_type_from_data())