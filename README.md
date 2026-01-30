# Credit Card Fraud Detection - ML

This project is a machine learning-based web application to detect fraudulent credit card transactions. It uses a Random Forest Classifier trained on a dataset of historical transactions to classify new transactions as either fraudulent or normal.

![Screenshot of the web application](https://i.imgur.com/your-screenshot-url.png)  <!-- Replace with a real screenshot URL -->

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Web Application](#running-the-web-application)
  - [Using the Application](#using-the-application)
- [Model](#model)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Retraining the Model](#retraining-the-model)
- [Files](#files)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
.
├── app.py                          # Flask web application
├── Credit_Card_Fraud_Detection-ML.ipynb # Jupyter notebook for model training
├── fraud_Detection.pkl             # Saved model (pickle)
├── Froud_detection.joblib          # Saved model (joblib)
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── static/
│   └── style.css                   # CSS for the web app
└── templates/
    └── index.html                  # HTML for the web app
```

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Web Application

To start the Flask server, run the following command in your terminal:

```bash
python app.py
```

The application will be accessible at `http://127.0.0.1:5000` in your web browser.

### Using the Application

1.  Open your web browser and navigate to `http://127.0.0.1:5000`.
2.  Fill in the transaction details in the form:
    *   **Time:** The number of seconds elapsed between this transaction and the first transaction in the dataset.
    *   **Amount:** The transaction amount.
    *   **V1-V28:** Anonymized features from the transaction data.
3.  Click the "Detect Fraud" button to get the prediction.
4.  The result will be displayed below the form, indicating whether the transaction is "Normal" or "Fraudulent".

## Model

### Dataset

The model was trained on a dataset containing credit card transactions made by European cardholders in September 2013. The dataset is highly unbalanced, with a very small percentage of fraudulent transactions.

-   **Source:** The dataset is from Kaggle, available [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).
-   **Features:** It contains 30 numerical features. `Time` and `Amount` are the only features that have not been transformed. The features `V1` through `V28` are the result of a PCA transformation to protect user identities and sensitive features.
-   **Target Variable:** The `Class` column indicates whether a transaction is fraudulent (1) or not (0).

### Training

The machine learning model is a **Random Forest Classifier** from the `scikit-learn` library. The complete training process is detailed in the Jupyter Notebook (`Credit_Card_Fraud_Detection-ML.ipynb`). The steps include:

1.  Loading and exploring the data.
2.  Splitting the data into training and testing sets (80/20 split).
3.  Training the Random Forest Classifier on the training data.
4.  Evaluating the model's performance on the test data.
5.  Saving the trained model using `joblib` and `pickle`.

### Evaluation

The model's performance was evaluated using the following metrics:

| Metric                          | Score  |
| ------------------------------- | ------ |
| Accuracy                        | 0.9996 |
| Precision                       | 0.9868 |
| Recall                          | 0.7653 |
| F1-Score                        | 0.8621 |
| Matthews Correlation Coefficient| 0.8689 |

Due to the imbalanced nature of the dataset, accuracy is not the most reliable metric. The high precision and strong F1-score indicate that the model is effective at identifying fraudulent transactions while minimizing false positives.

### Retraining the Model

To retrain the model with new or updated data, you can use the `Credit_Card_Fraud_Detection-ML.ipynb` notebook.

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open the notebook:**
    In the Jupyter interface in your browser, click on `Credit_Card_Fraud_Detection-ML.ipynb`.

3.  **Modify the data source (if necessary):**
    If you have a new dataset, update the file path in the notebook where the data is loaded:
    ```python
    # In the notebook, find this line and change the file path
    df = pd.read_csv("gfg_creditcard.csv")
    ```

4.  **Run the notebook:**
    You can run all the cells in the notebook by clicking on `Cell > Run All` in the menu. This will train a new model and save it as `Froud_detection.joblib` and `fraud_Detection.pkl` in the project's root directory.

## Files

-   `app.py`: The main Flask application that serves the web interface and makes predictions.
-   `Credit_Card_Fraud_Detection-ML.ipynb`: A Jupyter Notebook with the complete code for data analysis, model training, and evaluation.
-   `Froud_detection.joblib`: The trained model saved using `joblib`, which is loaded by the Flask app.
-   `requirements.txt`: A list of all Python libraries required to run the project.
-   `templates/index.html`: The HTML template for the web application's user interface.
-   `static/style.css`: The CSS file for styling the web application.

## Contributing

Contributions are welcome! If you have any suggestions or find any bugs, please open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (You may need to create a LICENSE file).
