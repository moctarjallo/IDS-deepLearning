import dash
import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Output, Event

import plotly
import plotly.graph_objs as go

import random 
from collections import deque

import pandas as pd

# # Read Data
# with open('../../data/kddcup.names.txt', 'r') as names_file:
#         lines = names_file.readlines()[1:]
#         names = [lines[i].split(':')[0] for i in range(len(lines))]
#         names.append('attack_type')
# df = pd.read_csv("../../data/kddcup.data_10_percent_corrected", names=names, nrows=20000)


from data import KddCupData
df = KddCupData()

X = deque(maxlen=50)
Y = deque(maxlen=50)
X.append(1)
Y.append(1)


# App
app = dash.Dash()

app.layout = html.Div(children=[
    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(id='graph-update', interval=1000)
])

@app.callback(
    Output('live-graph', 'figure'),
    events=[Event('graph-update', 'interval')]
)
def update_graph(properti):
    global X
    global Y
    global df
    X.append(X[-1]+1)
    Y.append(df[properti][X[-1]])
    data = go.Scatter(
        x = list(X), 
        y = list(Y), 
        name='scatter',
        mode='lines+markers'
    )

    return {'data': [data], 'layout': go.Layout(xaxis=dict(range=[min(X), max(X)]),
                                                yaxis=dict(range=[min(Y), max(Y)]))}

if __name__ == '__main__':
    app.run_server(debug=True)
    # print(data['duration'][216])