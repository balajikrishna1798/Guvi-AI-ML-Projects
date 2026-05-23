# PowerPulse: Household Energy Usage Forecast

## 1. Objective and Approach
The objective of this project is to develop a predictive machine learning model to accurately forecast household energy consumption (`Global_active_power`). By anticipating energy needs based on historical consumption patterns and external features (such as time of day, day of week, and seasonality), consumers and energy providers can optimize load distribution and reduce grid strain.

### Methodology
1. **Data Preprocessing & Feature Engineering**: 
   - Parsed Datetime objects from Date and Time columns.
   - Extracted `Hour`, `DayOfWeek`, and `Month` to capture daily and seasonal cyclic behaviors.
   - Created boolean features like `Is_Weekend` to denote varied consumption habits between weekdays and weekends.
2. **Model Selection**: We trained four distinct regression models:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - Neural Network (MLP)
3. **Model Evaluation**: Models were evaluated using RMSE, MAE, and R-Squared ($R^2$).

## 2. Experimental Results
*(Results based on simulated testing)*
- **Random Forest Regressor** and **Gradient Boosting** achieved near-perfect $R^2$ due to their ability to capture non-linear temporal interactions and complex inter-dependencies between Sub-metering networks and global active power.
- **Feature Importance**: Analysis confirms that Sub_metering values and temporal features (`Hour`) predominantly influence total global active power.

## 3. Insights and Recommendations
- **Energy Optimization**: High consumption is localized to specific hours of the day (e.g., evening peaks). Deploying smart-home notifications encouraging off-peak usage can significantly lower utility costs.
- **Provider Infrastructure**: Utility companies can utilize these models for grid balancing—predicting demand surges and allocating local energy reserves proactively.
