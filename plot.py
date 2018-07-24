import dash 
import dash_core_components as dcc 
import dash_html_components as html 

import pandas as pd

# Load the names of columns
with open('data/kddcup.names.txt', 'r') as names_file:
    lines = names_file.readlines()[1:]
    names = [lines[i].split(':')[0] for i in range(len(lines))]
names.append('attack_type')

# Load the data
data = pd.read_csv("data/kddcup.data_10_percent_corrected", names=names, nrows=20000)
# print(len(data))
# print(data.src_bytes)


app = dash.Dash()

app.layout = html.Div(children=[
    html.H2("Visualizing Data"),
    dcc.Graph(id='kdd-graph',
              figure={
                  'data': [
                      {'x': data.index, 'y': data['src_bytes'], 'type': 'line', 'name': 'src_bytes'},
                  ],
                  'layout': {
                      'title': 'kddcup data'
                  }
              })
])

if __name__ == '__main__':
    app.run_server()