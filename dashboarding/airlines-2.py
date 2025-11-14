# import required libraries
import pandas as pd
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

FILE_PATH='/Users/fpj/Development/python/ml-with-python/dashboarding/data/'
FILE_NAME='airline_data.csv'

# Read the airline data into pandas dataframe
airline_data =  pd.read_csv(FILE_PATH + FILE_NAME, 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})
# Create a dash application layout
app = dash.Dash(__name__)

# Get the layout of the application and adjust it.
# Create an outer division using html.Div and add title to the dashboard using html.H1 component
# Add a html.Div and core input text component
# Finally, add graph component.
app.layout = html.Div(children=[html.H1(),
                                html.Div(["Input Year", dcc.Input(),], 
                                style={}),
                                html.Br(),
                                html.Br(),
                                html.Div(),
                                ])

# Run the app
if __name__ == '__main__':
    app.run_server()