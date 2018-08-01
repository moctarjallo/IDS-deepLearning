import dash 
import dash_core_components as dcc 
import dash_html_components as html 

from dash.dependencies import Input, Output

import pandas as pd

# Load the names of columns


# Load the data
# print(len(data))
# print(data.src_bytes)

with open('data/kddcup.names.txt', 'r') as names_file:
        lines = names_file.readlines()[1:]
        names = [lines[i].split(':')[0] for i in range(len(lines))]
        names.append('attack_type')



app = dash.Dash()

app.layout = html.Div(children=[
    html.H2("Visualizing Data"),
    dcc.Input(id='kdd_input', type='text', value=''),
    html.Div(id='kdd_graph'),
])

@app.callback(
    Output(component_id='kdd_graph', component_property='children'),
    [Input(component_id='kdd_input', component_property='value')]
)
def update_graph(column):
    data = pd.read_csv("data/kddcup.data_10_percent_corrected", names=names, nrows=20000)

    graph = dcc.Graph(id='kdd_graph',
        figure={
            'data': [
                # {'x': data.index, 'y': data['attack_type'], 'name': 'attack_type'},
                {'x': data.index, 'y': data[column], 'name': column} for column in names
                # {'x': data['duration'], 'y': data['attack_type'], 'name': 'duration vs attack_type'}
            ], 
            'layout': {
                'title': column
            }
    })

    return graph


if __name__ == '__main__':
    app.run_server(debug=True) 