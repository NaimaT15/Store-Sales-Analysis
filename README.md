## Rossmann Store Sales Analysis 👋
## Overview
This project aims to predict sales in Rossmann stores across different cities six weeks ahead of time. The objective is to provide the Rossmann finance team with actionable insights and predictive models that can forecast sales based on factors such as promotions, holidays, competition, and other store-related features.

## Project Structure
The project is structured into the following tasks:

 ## 1 Exploration of Customer Purchasing Behavior:

- Conducted exploratory data analysis (EDA) to understand how promotions, store types, holidays, and other features impact sales.
- Visualized data trends and correlations to identify key factors influencing sales.

## 2 Prediction of Store Sales Using Machine Learning:

- Built machine learning models using scikit-learn.
- Processed the dataset to convert categorical features to numeric, handled missing values, and scaled the features for better prediction accuracy.
- Implemented a Random Forest Regressor model to predict sales based on historical data.

## 3 Building and Serving a Deep Learning Model:

- Developed a Long Short-Term Memory (LSTM) deep learning model to predict sales using time-series data.
- Applied LSTM to capture long-term dependencies in the sales data.
- Used TensorFlow and Keras libraries for building the LSTM model.
## 4 Model Serving with FastAPI:

- Created a REST API using FastAPI to serve the trained machine learning and deep learning models.
- Defined endpoints that accept input features and return sales predictions.

## Key Features

- **Data Preprocessing:** Handled missing data, converted categorical variables to numeric, and applied one-hot encoding to categorical features.
- **Machine Learning Models:** Built and trained Random Forest and LSTM models for regression analysis.
- **Deep Learning for Time Series:** Leveraged LSTM networks to predict future sales.
- **API for Predictions:** Deployed a REST API that delivers real-time sales predictions based on input data.
- **Logging & Serialization:** Logs important steps in the model pipeline and serialized trained models for version control.

## Tech Stack

- **Python:** The core programming language used for this project.
  
- **Libraries:**
- `Pandas`, `NumPy:` For data manipulation and preprocessing.
- `Scikit-learn:` For building and training the machine learning models.
- `TensorFlow`, `Keras:` For implementing LSTM and deep learning models.
- `FastAPI:` For creating and serving the REST API.
- `Matplotlib:` For data visualization.
- **Git & GitHub:** Version control and code collaboration.
- **Docker:** For containerizing the API and application.
## Installation
**1. Clone the repository**
```bash
git clone https://github.com/NaimaT15/Store-Sales-Analysis.git
cd Store-Sales-Analysis
```
**2. Set up the virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
**3. Install dependencies**
```bash
pip install -r requirements.txt
```
**4. Run the FastAPI server**
```bash
uvicorn main:app --reload
```
Access the FastAPI API documentation at `http://127.0.0.1:8000/docs.`

## Usage
**1. Exploratory Data Analysis (EDA)**

- The Jupyter notebook files for EDA and machine learning models can be found in the `notebooks/` folder.
- Use the `rossmann_store_sales_analysis.ipynb` for detailed analysis and visualizations.
  
**2. Machine Learning Model**

- The Random Forest Regressor model is built and trained in the `prediction_store_analysis.py script.`
- The serialized model is saved with a timestamp for reproducibility.
  
**3. Deep Learning Model**

- The LSTM deep learning model for time-series prediction is implemented in the `prediction_store_analysis.py` script under the `lstm_prediction function`.
  
**4. API for Sales Predictions**
  
- The FastAPI application accepts input data for features such as store type, promotions, and holidays and returns sales predictions.
- API endpoints are documented at `http://127.0.0.1:8000/docs.`
  
## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.

## Acknowledgements
- **10 Academy:** For providing the challenge.

## Contribution
Contributions are welcome! Please create a pull request or issue to suggest improvements or add new features.


## Author

👤 **Naima Tilahun**

* Github: [@NaimaT15](https://github.com/NaimaT15)

## Show your support

Give a ⭐️ if this project helped you!

