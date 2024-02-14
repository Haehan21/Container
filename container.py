# MODULES
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
import statsmodels
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX


# DATA PREPARATION
raw = pd.read_csv('container.csv')
data = raw[12:] # 딱 떨어지게 10년치 데이터로 
data.info()
train_df, test_df = train_test_split(data, test_size=0.1, shuffle=False)
train_df.index = data.date[:len(train_df)]
test_df.index = data.date[-len(test_df):]


# MODEL FITTING
opt_model = auto_arima(train_df['container'], 
                       start_p=0, start_q =0, max_p=3, max_q=3, max_d=2,
                       m=12, sesonal=True,
                    #    d=0, D=2, max_P=3, max_Q=3,
                       trace=True, error_action='warn', suppress_warnings=True, stepwise=True, random_state=20)

opt_model.summary()

prediction = opt_model.predict(len(test_df), return_conf_int=True) 
predicted_val = (prediction[0]) 
predict_index = test_df.index
pred_df = pd.DataFrame(predicted_val, predict_index, columns= ['container_pred'])


#  PREDICTION
model = SARIMAX(endog=data['container'],
            # exog=x_train
            order=(opt_model.order), seasonal_order=opt_model.seasonal_order,
            trend=None, measurement_error=True,
            time_varying_regression=True,
            mle_regression=False,
            simple_differencing=False,
            enforce_stationarity=True,
            enforce_invertibility=True,
            hamilton_representation=False,
            concentrate_scale=False,
            trend_offset=1,
            use_exact_diffuse=False,
            dates=None,
            freq=None,
            validate_specification=True)

model_fit = model.fit(maxiter = 300)

start_pred_len = len(data['container'])
end_pred_len = start_pred_len + 12 - 1
pred_df = pd.DataFrame(model_fit.predict(start=start_pred_len, end=end_pred_len))
pred_df.to_csv(f'container_prediction.csv', encoding='cp949', index=True)
