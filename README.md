Car Price Prediction (Machine Learning Project)

A clean, end-to-end machine learning project that predicts used car prices based on key features such as manufacturing year, mileage, and engine power.
Built with Python, scikit-learn, and includes a Streamlit demo.
car-price-prediction/
│
├── data/
│   └── car_data.csv         # Dataset (300 rows, ready to train)
│
├── src/
│   ├── train_model.py       # Training script (Random Forest)
│   └── demo_app.py          # Streamlit demo app
│
├── models/
│   └── rf_model.joblib      # Saved ML model (created after training)
│
├── notebooks/
│   └── .gitkeep             # (Optional) Jupyter notebooks will go here
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
Dataset Details

The dataset includes 300 synthetic car records with the following fields:
Column
Description
year
Year of manufacture
mileage
Total kilometers driven
power
Horsepower of the vehicle
price
Target variable – car selling price
Dataset file: data/car_data.csv

How to Run the Project

1.Install Dependencies
pip install -r requirements.txt

2. Train the Model
Run this script to train the Random Forest model:
python src/train_model.py
This will:
	•	Train the model
	•	Print RMSE & MAE
	•	Save the model as models/rf_model.jobli

3.Launch the Streamlit Demo App
streamlit run src/demo_app.py
The UI will ask for:
	•	age
	•	mileage
	•	power

And it will predict the car price

Model Information
	•	Algorithm: Random Forest Regressor
	•	Features: age, mileage, power
	•	Target: price
	•	Performance: RMSE & MAE displayed after training

  Skills Demonstrated

 Data preprocessing
 Feature engineering
 ML model design
 Model saving/loading
 Clean folder structure
 Streamlit app
 GitHub project struct

 License

MIT License — free to use and modify
  
