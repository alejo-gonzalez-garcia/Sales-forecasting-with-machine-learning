import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import darts.timeseries as ts 
import matplotlib.pyplot as plt

import darts.models as models
from darts import TimeSeries
from darts.utils.utils import ModelMode, SeasonalityMode
from sklearn.preprocessing import StandardScaler
import warnings
from sktime.forecasting.model_selection import SlidingWindowSplitter
from darts import TimeSeries
from sklearn.model_selection import GridSearchCV
from darts.models import LightGBMModel
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

def handle_outliers(data, detection_method=None, correction_method=None, contamination=0.05, interpolation_method='linear', **kwargs):
    """
    Handles outliers using specified detection and correction methods.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'standard_units'.
        detection_method (str): 'zscore', 'isolation_forest', or None.
        correction_method (str): 'median', 'interpolation', or None.
        contamination (float): Contamination level for Isolation Forest.
        interpolation_method (str): Interpolation method (e.g., 'linear', 'spline', 'pad').
        **kwargs: Additional parameters for interpolation methods.
    
    Returns:
        pd.DataFrame: DataFrame treated 'standard_units'.
    """
    if detection_method is None and correction_method is None:
        # No outlier handling, return data unchanged
        return data[['standard_units']]
    
    # Outlier Detection
    if detection_method == 'zscore':
        data['zscore'] = zscore(data['standard_units'])
        data['Outlier'] = abs(data['zscore']) > 2  # Z-score threshold
    elif detection_method == 'isolation_forest':
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = isolation_forest.fit_predict(data[['standard_units']])
        data['Outlier'] = outliers == -1
    else:
        data['Outlier'] = False  # No handling if detection_method is None or invalid
    
    # Outlier Correction
    if correction_method == 'median':
        median_value = data.loc[~data['Outlier'], 'standard_units'].median()
        data.loc[data['Outlier'], 'standard_units'] = median_value
    elif correction_method == 'interpolation':
        data.loc[data['Outlier'], 'standard_units'] = None
        # if interpolation_method == 'pad':
        #     data['standard_units'] = data['standard_units'].ffill().bfill()
        # else:
        data['standard_units'] = data['standard_units'].interpolate(method=interpolation_method, **kwargs)
        data['standard_units'] = data['standard_units'].bfill().ffill()  # Ensure no NaNs remain
    
    return data[['standard_units']]


def get_outlier_handling_params(country, molecule):
    """
    Retrieve the best outlier handling parameters for a given country and molecule.
    """
    best_params_df = pd.read_csv('best_outlier_handling.csv')

    params = best_params_df[
        (best_params_df['Country'] == country) & (best_params_df['Molecule'] == molecule)
    ]
    if not params.empty:
        # Extract "Outlier Handling" status
        outlier_handling = params['Outlier Handling'].values[0]

        if outlier_handling == 'None':
            # If no outlier handling is applied, return None for all parameters
            return None, None, None, None
        else:
            # Retrieve parameters when outlier handling is applied
            detection_method = params['Detection'].values[0]
            correction_method = params['Correction'].values[0]
            contamination = params['Contamination'].values[0] if not pd.isnull(params['Contamination'].values[0]) else None
            interpolation_method = params['Interpolation Method'].values[0] if not pd.isnull(params['Interpolation Method'].values[0]) else None
            
            return detection_method, correction_method, contamination, interpolation_method
    else:
        # Raise an error if no parameters are found
        raise ValueError(f"No parameters found for {country} and {molecule}")



def cross_validation_ML_Model(data_train, data_test, seasonal_periods, scaling=False,
                               model = models.ExponentialSmoothing, country = 'COUNTRY_A', molecule = "MOLECULE_A",
                               trend = ModelMode.ADDITIVE, seasonal=SeasonalityMode.ADDITIVE, cutoff_date_sma = '2022-12-31', num_months_sma=3, 
                               n_days_future=0):
    # Preprocessing for scaling if required
    
    # Drop 'dosage_form' and 'strength' columns
    data_train = data_train.drop(columns=['dosage_form', 'strength'])

    # Filter for COUNTRY_A and MOLECULE_A
    filtered_data_train = data_train[(data_train['country_name'] == country) & (data_train['molecule'] == molecule)]
    
    # Aggregate by date (index) and sum the 'standard_units'
    aggregated_data_train = filtered_data_train.groupby('date')['standard_units'].sum()
    aggregated_data_train = aggregated_data_train.reset_index()
 
    aggregated_data_train = aggregated_data_train.sort_values('date')

    aggregated_data_train.set_index('date', inplace=True)   # Set 'date' as the index
    aggregated_data_train.sort_index(inplace=True)

    if data_test is not None:
        data_test = data_test.drop(columns=['dosage_form', 'strength'])
        filtered_data_test = data_test[(data_test['country_name'] == country) & (data_test['molecule'] == molecule)]
        aggregated_data_test = filtered_data_test.groupby('date')['standard_units'].sum()
        aggregated_data_test = aggregated_data_test.reset_index()
        aggregated_data_test.set_index('date', inplace=True)    # Set 'date' as the index
        aggregated_data_test.sort_index(inplace=True)
    
    # Retrieve parameters
    detection_method, correction_method, contamination, interpolation_method = get_outlier_handling_params(country, molecule)

    # Prepare the data
    treated_data_train = handle_outliers(
        aggregated_data_train.copy(),
        detection_method=detection_method,
        correction_method=correction_method,
        contamination=contamination,
        interpolation_method=interpolation_method,
        order=3 if interpolation_method in ['polynomial', 'spline'] else None
    )
    
    if scaling:
        scaler = StandardScaler()
        data_train_scaled = scaler.fit_transform(treated_data_train)
        item = TimeSeries.from_series(pd.Series(data_train_scaled.flatten(), 
                                                index=treated_data_train.index))
    else:
        item = TimeSeries.from_series(treated_data_train)


    # Extract binary event columns (all columns starting with 'event_')
    # event_columns = [col for col in aggregated_data_train.columns if col.startswith('event_')]

    # Convert binary event columns into TimeSeries objects
    # past_covariates_train = TimeSeries.from_dataframe(aggregated_data_train[event_columns])
    # past_covariates_test = TimeSeries.from_dataframe(aggregated_data_test[event_columns])
    # if model == models.ARIMA:
    #     model_init = model(p=5, d=1, q=5, seasonal_order=(0, 0, 0, 7), trend=None, random_state=None, add_encoders=None)  # You can adjust these orders to experiment

    # if model == models.KalmanFilter:
    #     model_init = model(dim_x=299)  # Adjust as needed
    
    if model == models.LightGBMModel:
        model_init = model(
            lags=seasonal_periods,
            # lags_past_covariates=lags_model,  # Use the same lags for past covariates
            random_state=42,
            colsample_bytree=0.8,
            gamma=0.05,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=5,
            n_estimators=1500,
            reg_alpha=0,
            reg_lambda=1,
            subsample=0.8
        )
    if model == models.RandomForest:
        model_init = model(
            lags=seasonal_periods,
            # lags_past_covariates=lags_model,  # Use the same lags for past covariates
            random_state=42,
            n_estimators=100,
            max_depth=9,
        )

    if model == models.XGBModel:
        model_init = model(
            lags=seasonal_periods,
            # lags_past_covariates=lags_model,  # Use the same lags for past covariates
            random_state=42,
            n_estimators=1500,
            max_depth=5,
            learning_rate=0.05,
            reg_alpha=0,
            reg_lambda=1,
            colsample_bytree=0.8,
            subsample=0.8,
        )
    if model != models.LightGBMModel and model != models.RandomForest and model != models.XGBModel and model != models.ARIMA and model != models.Prophet and model != models.KalmanForecaster:
        model_init = models.ExponentialSmoothing(trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods, damped=True, random_state=42)  # Adjust as needed

    # lgbm_model.fit(series=item, past_covariates=past_covariates_train)
    model_init.fit(item)

    # Forecast for the future period
    if n_days_future==0:
        n_days_future = len(aggregated_data_test); 
    print("Months to predict:", n_days_future) # Obtener el número de días adicionales en `test`

    # Predict using the test past covariates
    # predicted_values = lgbm_model.predict(n=n_days_future, past_covariates=past_covariates_test, verbose=True)
    predicted_values = model_init.predict(n=n_days_future, verbose=True)
    if scaling:
        predicted_values = scaler.inverse_transform(predicted_values.values().reshape(-1, 1)).flatten()
    else:
        predicted_values = predicted_values.values().flatten()

    if model == "SMA":
        # Check if there is enough data for the specified window_months
        available_months = len(aggregated_data_train.loc[:cutoff_date_sma])
        actual_window_months = min(4, available_months)  # Use the smaller of 4 or available months

        # Perform the forecast with the determined window size
        predicted_values = iterative_sma_forecast_monthly(
            data=aggregated_data_train,
            start_date=cutoff_date_sma,
            num_months=n_days_future,  # Predict for the specified horizon (e.g., 3 months)
            window_months=actual_window_months  # Adjust if less than 4 months are available
        )

        
    # Prepare dates and true values for plotting
    if data_test is not None:
        extra_dates = aggregated_data_test.index[-n_days_future:]  # Get the last n_days_future dates
        true_values = aggregated_data_test.iloc[-n_days_future:, 0]  # Actual values for the last n_days_future days
    else:
    # Create a date range starting from the end of training data
        start_date = treated_data_train.index[-1] + pd.Timedelta(days=1)
        extra_dates = pd.date_range(start=start_date, periods=n_days_future, freq='M')  # Adjust 'freq' as needed (e.g., 'M' for months)        true_values = None
        
        true_values = None
        aggregated_data_test = None

    end_of_training = treated_data_train.index[-1]
    # print(f"Length of predicted_values: {len(predicted_values)}")
    # print(f"Length of extra_dates: {len(extra_dates)}")

    return extra_dates, true_values, predicted_values, end_of_training, aggregated_data_train, aggregated_data_test, treated_data_train




def plot_predictions_1_year(train_data, true_values, predicted_values,
                        extra_dates, end_of_training, seasonal_periods, model_name, country = 'COUNTRY_A', molecule = "MOLECULE_A", treated_data=None):
    # Plotting
    plt.figure(figsize=(12, 6))
    # Plot the training data
    plt.plot(train_data["standard_units"], label="Training Data", color="blue")

    if treated_data is not None:
        plt.plot(treated_data.index, treated_data["standard_units"], 
                 label="Treated Data (Outliers Handled)", color="purple", linestyle="--")
    
    if true_values is not None:
        # Plot the real values for the last 365 days
        plt.plot(true_values, label=f"Real Values (Last {seasonal_periods} Months)", color="green", alpha=0.25)

    if extra_dates is not None:
        # Plot the predicted values for the last 365 days
        plt.plot(extra_dates, predicted_values, label=f"Predicted Values (Last {seasonal_periods} Months)", color="orange")

    # Mark the end of training period
    plt.axvline(x=end_of_training, color='red', linestyle='--', label='End of Training')

    # Customize the plot
    plt.title(f"Training Data and Predicted Values for Last {seasonal_periods} Months ({country}-{molecule}) of {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()
    plt.show()


def plot_predictions_zoom(molecule, country, extra_dates, true_values, predicted_values, 
                          model_name, days_to_plot, n_months_future, one_plot=True,
                            trend = '', seasonality = '', ):

    if one_plot == False:

        plt.figure(figsize=(12, 6))

        if extra_dates is not None:
            plt.plot(extra_dates, predicted_values, label=f"Predictions {model_name}", color="orange")
        
        if true_values is not None:
            plt.plot(true_values, label="Actual Values", color="blue")

        plt.title(f"Comparison of Actual Values ​​vs Predictions. {n_months_future} Months ({country}, {molecule}) of {model_name} \n {trend}  {seasonality}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()

    plt.figure(figsize=(12, 6))
    
    # if true_values is not None:

    #     if extra_dates is not None:
    #         plt.plot(extra_dates[:days_to_plot], predicted_values[:days_to_plot], 
    #             label=f"Predictions {model_name}", color="orange")
        
    #     plt.plot(true_values[:days_to_plot], label="Actual Values", color="blue")
    #     plt.title(f"Comparison of Actual Values ​​vs Predictions. {seasonal_periods} Months ({molecule}, {country}) of {model_name} \n {trend}  {seasonality}")
    #     plt.axhline(y=min(true_values[:50]), color='gray', linestyle='--', linewidth=0.5)
    #     plt.xlabel("Fecha")
    #     plt.ylabel("Valor")
    #     plt.legend()
    #     plt.grid()
    #     plt.show()
    return

def compute_errors(true_values, predicted_values):
    
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        non_zero_mask = true_values != 0
        mape = np.mean(np.abs((true_values[non_zero_mask] - predicted_values[non_zero_mask]) / true_values[non_zero_mask])) * 100
    return mae, rmse, mape



def plot_errors(errors, trend_seasonality, country, molecule):
    # Validate and process the errors list based on its structure
    if trend_seasonality:
        try:
            # Include trend and seasonality in the labels
            labels = [f"{error[0]}-{error[1]}-{error[2]}" for error in errors]  # Trend-Seasonality-SP
            mae_values = [error[3] for error in errors]
            rmse_values = [error[4] for error in errors]
            mape_values = [error[5] for error in errors]
        except IndexError as e:
            raise ValueError(f"Errors list structure is incorrect for trend_seasonality=True: {e}")
    else:
        try:
            # Use seasonal periods only
            labels = [error[0] for error in errors]  # SP
            mae_values = [error[1] for error in errors]
            rmse_values = [error[2] for error in errors]
            mape_values = [error[3] for error in errors]
        except IndexError as e:
            raise ValueError(f"Errors list structure is incorrect for trend_seasonality=False: {e}")

    # Plot setup
    x = np.arange(len(labels))  # Label locations
    width = 0.2  # Bar width

    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width, mae_values, width, label='MAE')
    rects2 = ax.bar(x, rmse_values, width, label='RMSE')
    rects3 = ax.bar(x + width, mape_values, width, label='MAPE')

    # Set labels and title
    ax.set_xlabel('Seasonal Periods' if not trend_seasonality else 'Trend-Seasonality-Seasonal Periods')
    ax.set_ylabel('Error')
    ax.set_title(f'Errors by Seasonal Periods {country}-{molecule}' if not trend_seasonality else 'Errors by Trend, Seasonality, and Seasonal Periods')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    # Add value annotations
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{round(height, 2)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Adjust layout
    fig.tight_layout()
    plt.show()
    return

def evaluate_forecast_model(
    data_train, 
    data_test, 
    scaling, 
    model, 
    country, 
    molecule, 
    seasonal_periods, 
    model_name, 
    days_to_plot=100, 
    one_plot=True,
    show_error = False,
    n_months_future=0
):
    """
    Function to evaluate a forecasting model with cross-validation, generate plots, and compute errors.

    Parameters:
        data_train: Training dataset
        data_test: Test dataset
        lags_model: Number of lag features for the model
        scaling: Boolean indicating whether scaling is applied
        model: The model class to be used (e.g., LightGBMModel)
        country: Name of the country being analyzed
        molecule: Name of the molecule being analyzed
        seasonal_periods: Seasonal periods for the time series
        model_name: Name of the model for labeling plots
        days_to_plot: Number of days to plot in the zoomed-in plot (default 100)
        one_plot: Whether to plot all subplots on a single figure (default True)
    """
    # Run cross-validation and get results
    extra_dates, true_values, predicted_values, end_of_training, aggregated_data_train, aggregated_data_test, treated_data_train = cross_validation_ML_Model(
        data_train=data_train,
        data_test=data_test,
        seasonal_periods=seasonal_periods,
        scaling=scaling,
        model=model,
        country=country,
        molecule=molecule,
        n_days_future=n_months_future,
        cutoff_date_sma=data_test.index.min()
    )

    # Plot predictions for 1 year
    plot_predictions_1_year(
        train_data=aggregated_data_train,
        true_values=true_values,
        predicted_values=predicted_values,
        extra_dates=extra_dates,
        end_of_training=end_of_training,
        seasonal_periods=seasonal_periods,
        model_name=model_name,
        country=country,
        molecule=molecule,
        treated_data=treated_data_train
    )


    if one_plot == False:
        # Plot zoomed-in predictions
        plot_predictions_zoom(
            molecule=molecule,
            country=country,
            extra_dates=extra_dates,
            true_values=true_values,
            predicted_values=predicted_values,
            model_name=model_name,
            days_to_plot=days_to_plot,
            n_months_future=n_months_future,
            one_plot=one_plot
        )

    if data_test is not None:

        # Compute errors
        errors = []
        mae, rmse, mape = compute_errors(true_values, predicted_values)
        errors.append((model_name, mae, rmse, mape))

        if show_error:
            # Plot errors
            plot_errors(errors, trend_seasonality=False, country=country,
            molecule=molecule)
    else:
    
        errors = None

    # Return computed errors and key data for further analysis
    return {
        "errors": errors,
        "true_values": true_values,
        "predicted_values": predicted_values,
        "extra_dates": extra_dates,
        "end_of_training": end_of_training,
        "aggregated_data_train": aggregated_data_train,
        "aggregated_data_test": aggregated_data_test,
        "treated_data_train": treated_data_train
    }




def iterative_sma_forecast_monthly(data, start_date, num_months=12, window_months=4):
    """
    Iteratively forecast sales for a store using a Simple Moving Average (SMA) approach for monthly data.
    
    Parameters:
        data (pd.DataFrame): The dataset containing sales data for multiple stores.
                            Assumes the index is the first day of each month.
        start_date (str): The date to start predictions (e.g., '2015-01-01').
        num_months (int): Number of months to predict.
        window_months (int): Number of months to use for the SMA window.
    
    Returns:
        pd.Series: Predicted values for the specified number of months.
    """
    # Ensure the index is datetime for proper slicing
    data.index = pd.to_datetime(data.index)
    
    # Initialize the predictions list
    predictions = []
    
    # Identify the initial training period
    start_train = pd.to_datetime(start_date) - pd.DateOffset(months=window_months)
    start_train = pd.to_datetime(start_train)
    training_data = data.loc[start_train:start_date]
    
    # Iteratively forecast each month
    for month in range(num_months):
        # Check if there is sufficient data for the moving average
        if len(training_data) < window_months:
            raise ValueError("Insufficient data to perform forecast.")
        
        # Calculate the SMA for the next month
        predicted_month = training_data.iloc[-window_months:].mean()
        
        # Append the prediction for this month
        predictions.append(predicted_month)
        
        # Update training data with the new prediction
        next_month_date = pd.to_datetime(start_date) + pd.DateOffset(months=month)
        training_data.loc[next_month_date] = predicted_month
    # Create a date range for the predictions
    prediction_dates = pd.date_range(start=start_date, periods=num_months, freq='MS')  # 'MS' is Month Start
    return pd.Series(predictions, index=prediction_dates)


def ensemble_forecasts(errors_df, predictions_dict, true_values):
    """
    Combine the top 3 models using their mean predictions to create an ensemble forecast.
    Ensures predictions are aligned with the true_values.
    """

    # Rank models by MAE and get top 3 models
    errors_df = errors_df.sort_values(by='MAE')
    top_3_models = errors_df['Model'].iloc[:3].tolist()  # Select top 3 models
    print(f"Top 3 models: {top_3_models}")
    
    # Combine predictions from the top 3 models
    top_3_predictions = [
        predictions_dict[model] for model in top_3_models
    ]

    # Adjust lengths to match true_values
    for i, pred in enumerate(top_3_predictions):
        pred = pd.Series(pred, index=true_values.index)  # Ensure alignment with true_values index
        if len(pred) < len(true_values):
            extra_len = len(true_values) - len(pred)
            extended_values = pred.iloc[-1]  # Repeat the last value (could be improved)
            pred = pred.append(pd.Series([extended_values] * extra_len, index=true_values.index[-extra_len:]))
        elif len(pred) > len(true_values):
            # If the prediction is longer than the true values, trim the prediction
            pred = pred.iloc[:len(true_values)]  # Truncate extra predictions
        top_3_predictions[i] = pred

    # Calculate ensemble predictions as a weighted average
    weights = 1 / errors_df.loc[errors_df['Model'].isin(top_3_models), 'MAE']
    weights /= weights.sum()  # Normalize so weights sum to 1

    # Combine predictions using weighted average
    # weighted_predictions = np.average(np.array(top_3_predictions), axis=0, weights=weights)

    # Assign fixed weights to the top 3 models: 45%, 30%, and 25%
    weights = [0.45, 0.30, 0.25]

    # Combine predictions using weighted average
    weighted_predictions = np.average(np.array(top_3_predictions), axis=0, weights=weights)

    # Return the ensemble predictions as a Pandas Series with the same index as true_values
    return pd.Series(weighted_predictions, index=true_values.index), top_3_models

def best_model_selector(error_df):
    metrics = ['MAE', 'RMSE', 'MAPE']
    error_df['Model'] = error_df['Model'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    
    # Initialize the best_counts dictionary
    best_counts = {model: 0 for model in error_df['Model']}
    # Count the best metrics for each model
    for metric in metrics:
        best_value = error_df[metric].min()
        for i, row in error_df.iterrows():
            if row[metric] == best_value:
                best_counts[row['Model']] += 1

    # Add best metric counts to the DataFrame
    error_df['Best_Count'] = error_df['Model'].map(best_counts)

    # Check for models with at least 2 best metrics
    best_candidates = error_df[error_df['Best_Count'] >= 2]

    if len(best_candidates) == 1:
        return best_candidates.iloc[0]['Model']
    elif len(best_candidates) > 1:
        # If we have 3 models with 3 best metrics, we normalize and select the lowest one
        for metric in metrics:
            error_df[metric + '_norm'] = (error_df[metric] - error_df[metric].min()) / (error_df[metric].max() - error_df[metric].min())
        
        # Aggregate normalized scores
        error_df['Normalized_Score'] = error_df[[m + '_norm' for m in metrics]].mean(axis=1)

        # Choose the model with the lowest normalized score
        best_model_row = error_df.loc[error_df['Normalized_Score'].idxmin()]
        return best_model_row['Model']
    else:
        # If no model meets the criteria, fall back to lowest normalized score
        for metric in metrics:
            error_df[metric + '_norm'] = (error_df[metric] - error_df[metric].min()) / (error_df[metric].max() - error_df[metric].min())
        
        error_df['Normalized_Score'] = error_df[[m + '_norm' for m in metrics]].mean(axis=1)

        best_model_row = error_df.loc[error_df['Normalized_Score'].idxmin()]
        
        return best_model_row['Model']