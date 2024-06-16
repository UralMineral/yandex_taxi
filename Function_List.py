''' Function List:
        mean_absolute_percentage_error (y_true, y_pred)
        
        plotMovingAverage(series, window, 
                          plot_graph=False, plot_intervals=False, scale=1.96, plot_anomalies=False)
                          
        plotModelResults(model, X_train, y_train, X_test, y_test, window, 
                         plot_graph=False, plot_intervals=False, plot_anomalies=False)
                         
                         
        plotCoefficients(model, X_train)
        
        window_choice(window, model, test_size, 
                      add_features=False, plot_graph=False, show_lags=False)
                      
        train_test_split_arrays(X, y, test_size)
        
        window_choice_arr(window, model, test_size, 
                          plot_graph=False, add_features=False, show_lags=False)

                          
        error_estimator(models, fig_size)
        '''
# ALMAZ
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plotMovingAverage(series, window, 
                      plot_graph=False, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size
        plot_graph - whether plot or not
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies

    """
    rolling_mean = series.rolling(window=window,min_periods=1).mean()

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
    if plot_graph:        
        plt.figure(figsize=(15, 5))
        plt.title("Moving average\n window size = {}".format(window))
        plt.plot(rolling_mean, "g", label="Rolling mean trend")
        plt.plot(series[window:], label="Actual values",alpha=0.5)
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()
    
    return rolling_mean # Returns transformation of initial data.

def plotModelResults(model, X_train, y_train, X_test, y_test, window, 
                     plot_graph=False, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies

    """

    prediction = model.predict(X_test)
    
    error = mean_absolute_percentage_error(y_test, prediction)
    error_r2 = r2_score(prediction, y_test)
    
    if plot_graph:
        plt.figure(figsize=(15, 7))
        plt.plot(np.array(y_test.index),prediction, "g", label="prediction", linewidth=2.0)
        plt.plot(np.array(y_test.index),y_test.values, label="actual", alpha=0.5, linewidth=2.0)

        if plot_intervals:
            cv = cross_val_score(
                model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error"
            )
            mae = cv.mean() * (-1)
            deviation = cv.std()

            scale = 1.96
            lower = prediction - (mae + scale * deviation)
            upper = prediction + (mae + scale * deviation)

            plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
            plt.plot(upper, "r--", alpha=0.5)

            if plot_anomalies:
                anomalies = np.array([np.NaN] * len(y_test))
                anomalies[y_test < lower] = y_test[y_test < lower]
                anomalies[y_test > upper] = y_test[y_test > upper]
                plt.plot(anomalies, "o", markersize=10, label="Anomalies")

        plt.title("R2 score {0:.2f}".format(error_r2)+' and Mean absolute percentage error {0:.2f}%'.format(error)+ ' On window {}'.format(window))
        plt.legend(loc="best")
        plt.tight_layout()
        plt.grid(True)
    
    return error
    
def plotCoefficients(model, X_train):
    """
        Plots sorted coefficient values of the model
    """

    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind="bar")
    plt.grid(True, axis="y")
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles="dashed");

def window_choice(window, model, test_size, 
                  add_features=False, plot_graph=False, show_lags=False):
    
    ''' Function for smoothing data locally and make fitting, prediction and plot.'''
    
    data_rm = plotMovingAverage(data, window=window)
    
    # If we want to increase number of features:
    if add_features:        
        data_rm.index = pd.to_datetime(data_rm.index)
        data_rm["hour"] = data_rm.index.hour
        data_rm["weekday"] = data_rm.index.weekday
        data_rm["is_weekend"] = data_rm.weekday.isin([5, 6]) * 1    
    
    # Adding the lag of the target variable
    for i in range(1, 10): 
        data_rm["lag_{}".format(i)] = data_rm.y.shift(i)
        
    X_rm =  data_rm.dropna().drop(["y"], axis=1)
    y_rm =  data_rm.dropna().y
        
    X_train_rm, X_test_rm, y_train_rm, y_test_rm = train_test_split(X_rm, y_rm, test_size=test_size,\
                                                                   shuffle=False)
    model.fit(X_train_rm, y_train_rm)
    
    if plot_graph:
        plotMovingAverage(data, window=window, plot_graph=True)
        error = plotModelResults(model, X_train_rm, y_train_rm,\
                                            X_test_rm, y_test_rm, window=window, plot_graph=True)
        if show_lags:
                plotCoefficients(model, X_train_rm)
    else:
        error = plotModelResults(model, X_train_rm, y_train_rm,\
                                           X_test_rm, y_test_rm, window=window)
        return error

def train_test_split_arrays(X, y, test_size):
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    
    test_index = int(len(X[0]) * (1 - test_size))
    
    X_train = [x.iloc[:test_index] for x in X]
    X_test = [x.iloc[test_index:] for x in X]
    y_train = [Y.iloc[:test_index] for Y in y]
    y_test = [Y.iloc[test_index:] for Y in y]

    return X_train, X_test, y_train, y_test

def window_choice_arr(window, model, test_size, 
                      plot_graph=False, add_features=False, show_lags=False):
    
    data_rms = []
    X_rms = []
    y_rms = []
    
    for i in range(1, window+1):
        data_rms.append(plotMovingAverage(data, window=i))
    
    if add_features:        
        for data_rm in data_rms:
            data_rm.index = pd.to_datetime(data_rm.index)
            data_rm["hour"] = data_rm.index.hour
            data_rm["weekday"] = data_rm.index.weekday
            data_rm["is_weekend"] = data_rm.weekday.isin([5, 6]) * 1    
    
    # Adding the lag of the target variable
    for data_rm in data_rms:
        for i in range(1, 10): 
            data_rm["lag_{}".format(i)] = data_rm.y.shift(i)
        X_rms.append(data_rm.dropna().drop(["y"], axis=1))
        y_rms.append(data_rm.dropna().y)
        
    X_train_rms, X_test_rms, y_train_rms, y_test_rms = train_test_split_arrays(X_rms, y_rms, test_size)

    model_array = []
    errors = []
    
    for i, data_rm in enumerate(data_rms):
        model_array.append(model.fit(X_train_rms[i], y_train_rms[i]))
        if plot_graph:
            errors.append(plotModelResults(model_array[i], X_train_rms[i], y_train_rms[i],\
                                            X_test_rms[i], y_test_rms[i], i+1, plot_graph=True))
            if show_lags:
                plotCoefficients(model_array[i], X_train_rms[i])
        else:
            errors.append(plotModelResults(model_array[i], X_train_rms[i], y_train_rms[i],\
                                            X_test_rms[i], y_test_rms[i], i+1))
    return errors

def error_estimator(models, fig_size):
    
    ''' Funcition plots same as in above plot, 
        where we analyse which model is the best decision for our dataset (no preprocessing and tuning (yet))'''
    
    num_models = len(models)
    fig, axs = plt.subplots(num_models, num_models, figsize=(fig_size, fig_size))
    
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i != j:
                ax = axs[i, j]
                err_i = window_choice_arr(10, model_i, 0.2)
                err_j = window_choice_arr(10, model_j, 0.2)
                
                ax.plot(np.array(err_i[1:]) - np.array(err_j[1:]))
                ax.set_xlabel('window')
                ax.set_ylabel('err_diff in %'+" = {0:.2f} %".format(np.mean(np.array(err_i[1:]) - np.array(err_j[1:]))))
                ax.set_title(f'{model_i} - {model_j}')
                ax.grid(True)
    
    # Hide empty subplots
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                axs[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

