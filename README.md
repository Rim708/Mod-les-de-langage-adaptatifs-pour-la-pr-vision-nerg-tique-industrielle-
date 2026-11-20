# Energy Consumption Forecasting Using Deep Learning

## Overview

This project focuses on predicting energy consumption and thermal loads using Deep Learning techniques and time series analysis.

The objective is to develop forecasting models capable of predicting three energy-related variables simultaneously:

- **Electricity Consumption (KW):** Electrical power demand.
- **Cooling Load (CHWTON):** Cooling demand measured in Chilled Water Tons.
- **Heating Load (HTmmBTU):** Thermal heating demand measured in million British Thermal Units.

Instead of building three independent forecasting models, the project explores a **multi-output learning approach**, allowing a single model to learn relationships between electricity consumption, cooling demand, and heating demand.

Accurate energy forecasting can help improve energy efficiency, reduce operational costs, anticipate future demand, and support decision-making in smart buildings and industrial environments.

## Dataset

The project uses an hourly energy consumption dataset containing **17,520 observations and 21 columns**, covering two years of data (2018–2019).

The dataset includes:

### Energy-related variables
- `KW`: Electricity consumption.
- `CHWTON`: Cooling load.
- `CHWTON(KW)`: Cooling load converted to kilowatts.
- `HTmmBTU`: Heating load.
- `HTmmBTU(KW)`: Heating load converted to kilowatts.
- `Combined mmBTU`: Combined thermal energy.
- `Combined mmBTU(KW)`: Combined energy converted to kilowatts.
- Power factor variables: `KW_PF`, `CHWTON_PF`, `HTmmBTU_PF`.

### Weather variables
- Direct Horizontal Irradiance (DHI).
- Direct Normal Irradiance (DNI).
- Global Horizontal Irradiance (GHI).
- Dew Point.
- Wind Speed.
- Precipitable Water.
- Wind Direction.
- Relative Humidity.
- Temperature.
- Pressure.

The weather variables provide additional information that can help the models understand the relationship between environmental conditions and energy demand.

## Methodology

The project follows several steps:

### 1. Data Preparation
- Load the dataset using Pandas.
- Convert the `datetime` column into a datetime format.
- Set the datetime column as the index.
- Sort observations chronologically.
- Inspect data types and statistical distributions.

### 2. Feature Engineering

The project explores temporal features such as:
- Hour of the day.
- Day of the week.
- Month.

These features help capture daily, weekly, and seasonal patterns in energy consumption.

### 3. Data Normalization

`MinMaxScaler` and `StandardScaler` are explored to normalize numerical variables and improve model training.

### 4. Time Series Sequence Generation

The forecasting problem is transformed into a supervised learning problem using sliding windows.

The model uses the previous 24 hours of observations to predict the next hour.

- Input sequence: 24 time steps.
- Input features: 20 variables.
- Prediction horizon: 1 hour.
- Output variables: 3 energy-related targets.

The resulting input and output shapes are:

- **X:** `(17495, 24, 20)`
- **y:** `(17495, 1, 3)`

### 5. Dataset Splitting

A chronological split is used to preserve the temporal order of observations:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

This allows model evaluation on future observations that were not used during training.

## Deep Learning Models

The notebook explores recurrent and temporal deep learning architectures for time series forecasting, including:

- **LSTM (Long Short-Term Memory):** Captures long-term dependencies in sequential data.
- **GRU (Gated Recurrent Unit):** A recurrent architecture with a simpler gating mechanism.
- **Temporal Convolutional Networks (TCN):** Uses temporal convolutions to learn patterns across time.
- **Attention Mechanisms:** Explores the importance of different time steps in sequential predictions.

The experiments use both TensorFlow/Keras and PyTorch.

The multi-output forecasting approach aims to learn shared patterns between electricity consumption, cooling demand, and heating demand.

## Model Training

The notebook includes experiments with:

- Adam optimizer.
- Mean Squared Error and custom weighted loss functions.
- Batch-based training.
- Training and validation loss monitoring.
- Gradient clipping in PyTorch experiments.
- Best-model checkpointing based on validation loss.

Training parameters are adjusted across experiments.

## Evaluation Metrics

Model performance is evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between predictions and actual values.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors.
- **MAPE (Mean Absolute Percentage Error):** Measures prediction error as a percentage.

The notebook also includes visual comparisons between actual and predicted values.

### Reported Evaluation Results

One of the multi-output forecasting experiments reports the following results:

| Target | MAE | RMSE |
|---|---:|---:|
| Electricity (KW) | 452.82 | 656.39 |
| Cooling (CHWTON(KW)) | 689.87 | 983.89 |
| Heating (HTmmBTU(KW)) | 77.22 | 103.69 |

Reported MAPE values:

| Target | MAPE |
|---|---:|
| Electricity | 2.28% |
| Cooling | 3.41% |
| Heating | 3.37% |

These values correspond to the evaluated experiment in the notebook and should not be interpreted as results for every explored architecture.

## Technologies Used

- **Python**
- **Pandas & NumPy** – Data manipulation and numerical computing.
- **Matplotlib & Seaborn** – Data visualization.
- **Scikit-learn** – Data preprocessing and evaluation metrics.
- **TensorFlow / Keras** – Deep Learning model development.
- **PyTorch** – Neural network implementation and training.
- **LSTM / GRU / TCN** – Time series forecasting architectures.

## Project Structure

```text
Energy-Consumption-Forecasting/
│
├── data/
│   └── data_set_adjusted.csv
│
├── notebooks/
│   └── energy_forecasting.ipynb
│
├── models/
│   └── saved_models/
│
├── README.md
└── requirements.txt
```

*This is a suggested repository organization.*

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Energy-Consumption-Forecasting.git

cd Energy-Consumption-Forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place `data_set_adjusted.csv` inside the `data/` directory.

Update the dataset path in the notebook if necessary.

### 4. Run the notebook

```bash
jupyter notebook
```

Open `energy_forecasting.ipynb` and execute the cells to explore the data preparation, model training, predictions, and evaluation process.

## Potential Applications

- Energy demand forecasting.
- Smart building energy management.
- Cooling and heating load prediction.
- Energy efficiency optimization.
- Operational cost reduction.
- Predictive maintenance planning.

## Conclusion

This project explores the use of Deep Learning for multi-output energy forecasting.

By combining historical energy consumption, weather conditions, and temporal information, the developed approaches aim to capture complex relationships between different energy demands.

The project provides a foundation for intelligent energy management systems and data-driven optimization in industrial and smart building environments.
