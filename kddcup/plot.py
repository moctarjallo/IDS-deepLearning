
import dash
from dash.dependencies import Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import numpy as np

from core.data import KddCupData
df = next(KddCupData(filename='data/corrected', nrows=1000))

X = deque(maxlen=20)
X.append(0)
Y = deque(maxlen=20)
Ys = []
for i in range(41):
    Ys.append(Y.copy())
# Y.append(0)


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),
    ]
)

@app.callback(Output('live-graph', 'figure'),
              events=[Event('graph-update', 'interval')])
def update_graph_scatter():
    X.append(X[-1]+1)
    y = df[['dst_bytes', 'src_bytes']].X
    Ys[0].append(y.loc[X[-1]][1])

    data = [plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Ys[0]),
            name='Scatter',
            mode= 'lines+markers'
            )]

    return {'data': data,
            'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                 yaxis=dict(range=[min(Ys[0]),max(Ys[0])]),)}



if __name__ == '__main__':
    app.run_server(debug=True)