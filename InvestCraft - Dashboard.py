# # InvestCraft - Dashboard

# ## Content
# * [1. Loading the Libraries and the data](#1)
# * [2. Code for the dashboard Interface](#2)
# * [3. Code for the underlying functions within the interface](#3)
# 
# 
# #### Note that the dashboard opens up in a separate browser. The url for the browser will be produced in the end of the code and would look something like "http://127.0.0.1:8080"

# <a id='1'></a>
# ## 1. Loading the Libraries and the data

import pkg_resources
installedPackages = {pkg.key for pkg in pkg_resources.working_set}
required = {'dash', 'dash-core-components', 'dash-html-components', 'dash-daq', 'cvxopt' }
missing = required - installedPackages

# Importing the packages needed

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pickle import load
import cvxopt as opt
from cvxopt import solvers
from datetime import datetime, timedelta
import plotly.graph_objects as go
from langchain.chat_models import ChatOpenAI
from sklearn.svm import SVR


# df.head()
investors = pd.read_csv('InputData.csv', index_col = 0 )
investors.head(1)

# ### Load the market data and clean the data

assets = pd.read_csv('SP500Data.csv',index_col=0)
missing_fractions = assets.isnull().mean().sort_values(ascending=False)

missing_fractions.head(10)

drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

assets.drop(labels=drop_list, axis=1, inplace=True)
assets.shape
# Fill the missing values with the last value available in the dataset. 
assets=assets.fillna(method='ffill')
assets.head(2)

options=np.array(assets.columns)
# str(options)
options = []

for tic in assets.columns:
    #{'label': 'user sees', 'value': 'script sees'}
    mydict = {}
    mydict['label'] = tic #Apple Co. AAPL
    mydict['value'] = tic
    options.append(mydict)


# <a id='2'></a>
# ## 2. Code for the dashboard Interface

app = dash.Dash(__name__, external_stylesheets=['https://pcloud.codeestro.com/assets/css/tailwind.min.css'])

app.layout = html.Div([
    html.Div(className="bg-gray-100", children=[ 
        html.Section(className="flex flex-col md:flex-row h-screen items-center", children=[
            html.Div(className="bg-white w-full md:max-w-md lg:max-w-full md:mx-auto md:mx-0 md:w-1/2 xl:w-1/3 h-screen px-6 lg:px-16 xl:px-12 flex items-center justify-center", children=[
                html.Div(className="w-full h-100", children=[
                    html.H1(className="text-3xl md:text-4xl font-bold", children="InvestCraft - Dashboard"),
                    html.H2(className="text-2xl md:text-3xl font-bold leading-tight mt-12", children="Step 1: Enter Investor Characteristics"),
                    html.Div(className="mt-6", children=[
                        html.Div([
                            html.Label(className="block text-gray-700", children="Age:"),
                            dcc.Slider(
                                min=20,
                                max=70,
                                value=25,
                                className="w-full px-4 py-3 rounded-lg bg-gray-200 mt-2 border focus:border-blue-500 focus:bg-white focus:outline-none",
                                id="Age"
                            ),
                        ]),
                        html.Div([
                            html.Label(className="block text-gray-700", children="NetWorth:"),
                            dcc.Slider(
                                min=-1000000,
                                max=3000000,
                                value=10000,
                                marks={-1000000: '-₹1M', 0: '0', 500000: '₹500K', 1000000: '₹1M', 2000000: '₹2M'},
                                className="w-full px-4 py-3 rounded-lg bg-gray-200 mt-2 border focus:border-blue-500 focus:bg-white focus:outline-none",
                                id="Nwcat"
                            ),
                        ]),
                        html.Div([
                            html.Label(className="block text-gray-700", children="Income:"),
                            dcc.Slider(
                                min=-1000000,
                                max=3000000,
                                value=100000,
                                marks={-1000000: '-₹1M', 0: '0', 500000: '₹500K', 1000000: '₹1M', 2000000: '₹2M'},
                                className="w-full px-4 py-3 rounded-lg bg-gray-200 mt-2 border focus:border-blue-500 focus:bg-white focus:outline-none",
                                id="Inccl"
                            ),
                        ]),
                        html.Div([
                            html.Label(className="block text-gray-700", children="Education Level:"),
                            dcc.Slider(
                                min=1,
                                max=4,
                                value=2,
                                marks={1: 'No school', 2: 'High school', 3: 'College', 4: 'PHD'},
                                className="w-full px-4 py-3 rounded-lg bg-gray-200 mt-2 border focus:border-blue-500 focus:bg-white focus:outline-none",
                                id="Edu"
                            ),
                        ]),
                        html.Div([
                            html.Label(className="block text-gray-700", children="Married:"),
                            dcc.Slider(
                                min=1,
                                max=2,
                                value=1,
                                marks={1: 'Unmarried', 2: 'Married'},
                                className="w-full px-4 py-3 rounded-lg bg-gray-200 mt-2 border focus:border-blue-500 focus:bg-white focus:outline-none",
                                id="Married"
                            ),
                        ]),
                        html.Div([
                            html.Label(className="block text-gray-700", children="Kids:"),
                            dcc.Slider(
                                min=investors['KIDS07'].min(),
                                max=investors['KIDS07'].max(),
                                marks=[{'label': j, 'value': j} for j in investors['KIDS07'].unique()],
                                value=3,
                                className="w-full px-4 py-3 rounded-lg bg-gray-200 mt-2 border focus:border-blue-500 focus:bg-white focus:outline-none",
                                id="Kids"
                            ),
                        ]),
                        html.Div([
                            html.Label(className="block text-gray-700", children="Occupation:"),
                            dcc.Slider(
                                min=investors['OCCAT107'].min(),
                                max=investors['OCCAT107'].max(),
                                marks={1: 'Managerial', 2: 'Professional', 3: 'Sales', 4: 'Unemployed'},
                                value=3,
                                className="w-full px-4 py-3 rounded-lg bg-gray-200 mt-2 border focus:border-blue-500 focus:bg-white focus:outline-none",
                                id="Occ"
                            ),
                        ]),
                        html.Div([
                            html.Label(className="block text-gray-700", children="Willingness to take Risk:"),
                            dcc.Slider(
                                min=investors['RISK07'].min(),
                                max=investors['RISK07'].max(),
                                marks={1: 'Low', 2: 'Medium', 3: 'High', 4: 'Extreme'},
                                value=3,
                                className="w-full px-4 py-3 rounded-lg bg-gray-200 mt-2 border focus:border-blue-500 focus:bg-white focus:outline-none",
                                id="Risk"
                            ),
                        ]),
                        html.Button(
                            className="w-full block bg-blue-500 hover:bg-blue-400 focus:bg-blue-400 text-white font-semibold rounded-lg px-4 py-3 mt-6",
                            children="Calculate Risk Tolerance",
                            id="investor_char_button",
                            n_clicks=0
                        )
                    ])
                ])
            ]),
            html.Div(className="hidden lg:block w-full md:w-1/2 xl:w-2/3 h-screen", children=[
                html.Div(className="bg-white w-full flex items-center justify-center", children=[
                    html.Div(className="w-full h-100", children=[
                        html.H2(className="text-2xl md:text-3xl font-bold leading-tight mt-12", children="Step 2: Asset Allocation and Portfolio Performance"),
                        html.Div(className="mt-6", children=[
                            html.Div([
                                html.Label(className="block text-gray-700", children="Risk Tolerance (scale of 100):"),
                                dcc.Input(
                                    type="text",
                                    className="w-full px-4 py-3 rounded-lg bg-gray-200 mt-2 border focus:border-blue-500 focus:bg-white focus:outline-none",
                                    autoFocus=True,
                                    required=True,
                                    id="risk-tolerance-text",
                                    disabled=True
                                ),
                            ]),
                            html.Div([
                                html.Label(className="block text-gray-700", children="Select the assets for the portfolio:"),
                                dcc.Dropdown(
                                    id="ticker_symbol",
                                    options=options,
                                    value=['GOOGL', 'FB', 'LNT', 'IBM', 'AMZN', 'MSI'],
                                    multi=True,
                                    # style={'fontSize': 24, 'width': 75}
                                ),
                                html.Button(
                                    className="w-full block bg-blue-500 hover:bg-blue-400 focus:bg-blue-400 text-white font-semibold rounded-lg px-4 py-3 mt-6",
                                    children="Submit",
                                    id="submit-asset_alloc_button",
                                    n_clicks=0
                                )
                            ]),
                            html.Div([
                                dcc.Dropdown(
                                    id='time-interval-selector',
                                    options=[
                                        {'label': 'All Time', 'value': 'all_time'},
                                        {'label': 'Last Year', 'value': 'year'},
                                        {'label': 'Last 6 Months', 'value': '6_months'},
                                        {'label': 'Last Quarter', 'value': 'quarter'},
                                        {'label': 'Last Week', 'value': 'week'}
                                    ],
                                    value='all_time',
                                    className="w-full block rounded-lg px-4 py-3 mt-6",

                                )
                            ]),
                            html.Div([
                                dcc.Graph(
                                    id='Performance',
                                    style={'width': '100%', 'height': '100%'}
                                )
                            ], style={'width': '100%', 'height': '60vh', 'vertical-align': 'top', 'display': 'inline-block', \
                                      'font-family': 'calibri', 'horizontal-align': 'right'}),
                            html.Div([
                                dcc.Graph(
                                    id='Asset-Allocation',
                                    style={'width': '100%', 'height': '100%'}
                                ), 
                            ], style={'width': '50%', 'vertical-align': 'top', 'display': 'inline-block', \
                                      'font-family': 'calibri', 'horizontal-align': 'right'}),
                            html.Div([
                                html.Label(className="block text-gray-700", children="Investment Suggestion:"),
                                html.Div(id="investment-suggestion"),
                                html.Div(id="predicted-price-text")
                            ], style={'width': '50%', 'vertical-align': 'top', 'display': 'inline-block', \
                                      'font-family': 'calibri', 'horizontal-align': 'right'}),
                        ])
                    ])
                ])
            ])
        ])
    ])
])

# <a id='3'></a>
# ## 3. Code for the underlying functions within the interface
# 
# The steps performed are as follows: 
# 
# 1) Loading the regression model for predicting risk tolerance
# 
# 2) Using markovitz mean variance analysis for asset allocation
# 
# 3) Producing chart for the asset allocation and portfolio performance
# 
# #### Click the url produced by this code to see the dashboard

def predict_riskTolerance(X_input):

    filename = 'finalized_model.sav'
    loaded_model = load(open(filename, 'rb'))
    # estimate accuracy on validation set
    predictions = loaded_model.predict(X_input)
    return predictions

#Asset allocation given the Return, variance
def get_asset_allocation(riskTolerance,stock_ticker):
    #ipdb.set_trace()   
    assets_selected = assets.loc[:,stock_ticker]
    return_vec = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vec)
    returns = np.asmatrix(return_vec)
    mus = 1-riskTolerance
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(return_vec))
    pbar = opt.matrix(np.mean(return_vec, axis=1))
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    w=portfolios['x'].T
    print (w)
    Alloc =  pd.DataFrame(data = np.array(portfolios['x']),index = assets_selected.columns)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    returns_final=(np.array(assets_selected) * np.array(w))
    returns_sum = np.sum(returns_final,axis =1)
    returns_sum_pd = pd.DataFrame(returns_sum, index = assets.index )
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0,:] + 100   
    return Alloc,returns_sum_pd

    
def get_investment_suggestion(input_text):
    try:
        # Create an instance of ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key="your-api-key-here")
        
        # Prepare the messages for the model
        messages = [("human", input_text)]

        # Invoke the model with the input text
        response = llm.invoke(messages)

        # Extract and return the response text
        return response.content

    except Exception as e:
        return f"Error: {e}"

#Callback for the graph
#This function takes all the inputs and computes the cluster and the risk tolerance

# Update the callback function to include GPT-3 suggestion
@app.callback(
    [Output('risk-tolerance-text', 'value'), Output('investment-suggestion', 'children')],
    [Input('investor_char_button', 'n_clicks'),
     Input('Age', 'value'), Input('Nwcat', 'value'),
     Input('Inccl', 'value'), Input('Risk', 'value'),
     Input('Edu', 'value'), Input('Married', 'value'),
     Input('Kids', 'value'), Input('Occ', 'value')])

def update_risk_tolerance_and_suggestion(n_clicks, Age, Nwcat, Inccl, Risk, Edu, Married, Kids, Occ):
    RiskTolerance = 0
    suggestion = ""
    if n_clicks is not None:
        X_input = [[Age, Edu, Married, Kids, Occ, Inccl, Risk, Nwcat]]
        RiskTolerance = predict_riskTolerance(X_input)[0]
        input_text = f"For an investor with Age: {Age}, Education Level: {Edu} here are level means(1: No school, 2: High school, 3: College, 4: PHD), Married: {Married} this is the meaning of 1 and 2 (1: Unmarried, 2: Married), Kids: {Kids}, Occupation: {Occ} here is the meaning (1: Managerial, 2: Professional, 3: Sales, 4: Unemployed), Income: {Inccl} here is the meaning (1: Low, 2: Medium, 3: High, 4: Extreme), Risk Tolerance: {Risk}, Net Worth: {Nwcat}, the suggested investment allocation is... give proper detailed suggestions. just like you are an investment advisor."
        # suggestion = get_investment_suggestion(input_text)
    return round(float(RiskTolerance * 100), 2), suggestion

# Function to predict future prices for all days within the time interval
def predict_future_prices(InvestmentReturn_filtered, time_interval):
    # Extracting date and price data from InvestmentReturn_filtered
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in InvestmentReturn_filtered.index]
    prices = InvestmentReturn_filtered.iloc[:, 0].values.reshape(-1, 1)

    # Convert dates to numerical values (days since the start)
    start_date = min(dates)
    numerical_dates = [(date - start_date).days for date in dates]

    model = SVR()
    model.fit(np.array(numerical_dates).reshape(-1, 1), prices)

    # Calculate the future date based on the last available date in the dataset
    last_date = max(dates)
    future_dates = [last_date + timedelta(days=i) for i in range(1, 2*365)]  # Predict for 2 years (365 * 2 days)

    # Convert future dates to numerical values
    future_numerical_dates = [(date - start_date).days for date in future_dates]

    # Predict the prices for future dates
    predicted_prices = model.predict(np.array(future_numerical_dates).reshape(-1, 1))

    volatility = 0.03
    mean_return = 0.001
    for i in range(1, len(predicted_prices)):
        # Calculate the percentage change based on the previous price
        change = np.random.normal(loc=mean_return, scale=volatility)
        predicted_prices[i] = predicted_prices[i-1] + (predicted_prices[i-1] * change)

    return future_dates, predicted_prices


@app.callback([Output('Asset-Allocation', 'figure'),
               Output('Performance', 'figure'),
               Output('predicted-price-text', 'children')],
              [Input('submit-asset_alloc_button', 'n_clicks'),
               Input('risk-tolerance-text', 'value'),
               Input('time-interval-selector', 'value')], 
              [State('ticker_symbol', 'value')])
def update_asset_allocationChart(n_clicks ,risk_tolerance, time_interval, stock_ticker):
    if time_interval == 'year':
        interval_days = 365
    elif time_interval == '6_months':
        interval_days = 180
    elif time_interval == 'quarter':
        interval_days = 120
    elif time_interval == 'week':
        interval_days = 7
    else:  # 'all_time' option
        interval_days = None

    Allocated, InvestmentReturn = get_asset_allocation(risk_tolerance, stock_ticker)  

    # Filter InvestmentReturn data based on the selected time interval
    if interval_days:
        InvestmentReturn_filtered = InvestmentReturn.iloc[-interval_days:]
    else:
        InvestmentReturn_filtered = InvestmentReturn

    # Predict future prices for all days starting from the end of the actual data point and extending two years into the future
    future_dates, predicted_prices = predict_future_prices(InvestmentReturn_filtered, time_interval)

    # Create a trace for the predicted prices
    predicted_prices_data = go.Scatter(
        x=future_dates,
        y=predicted_prices.flatten(),
        mode='lines',
        name='Predicted Prices',
        marker=dict(color='blue'),
        hoverinfo='text',
        hovertext=[f'Predicted Price: ₹{price:.2f}, Date: {date.strftime("%Y-%m-%d")}' for date, price in zip(future_dates, predicted_prices.flatten())]
    )

    # Calculate the percentage change for all available points
    initial_investment = 100
    percentage_changes = ((InvestmentReturn_filtered - initial_investment) / initial_investment) * 100

    return [{'data': [go.Bar(
                        x=Allocated.index,
                        y=Allocated.iloc[:, 0],
                        marker=dict(color='red'),
                    )],
             'layout': {'title': "Asset allocation - Mean-Variance Allocation"}},
            {'data': [go.Scatter(
                        x=InvestmentReturn_filtered.index,
                        y=InvestmentReturn_filtered.iloc[:, 0],
                        mode='lines',
                        name='(%)',
                        marker=dict(color='red'),
                        hoverinfo='text',
                        hovertext=[f'Percentage Change: {change:.2f}%' for change in percentage_changes.iloc[:, 0]]
                    ), predicted_prices_data],
             'layout': {'title': "Portfolio value of ₹100 investment"}},
            f"Predicted Prices for the next 2 years shown in blue."]

if __name__ == '__main__':
    app.run_server()