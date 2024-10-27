import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import numpy as np
import plotly.graph_objects as go
import pandas as pd

import matplotlib.pyplot as plt
import json

from helpers import fill_template, read_config, read_json, concat_output
from inference import KnnClassifier


config = read_config()
model = KnnClassifier( visualization_method = "tsne", config = config )
model.embedder.get_embeddings ## Compute the roles embeddings if not already in embeddings.txt file.
df = model.embedder.get_initial_corrdinates()
X = df[["x","y","z"]].values
labels = df["role_id"].values
titles = pd.read_csv("./data/roles_all_w_intern_wo_admin_w_title_wo_legal.csv", index_col=0)["Title"].values

rol_to_id_dict = read_json("./data/role_to_id.json")
id_to_rol_dict =  dict((v,k) for k,v in rol_to_id_dict.items())

with open('./data/suggestions.json', 'r') as f:
    form_dict = json.load(f)

## create a Div for each dropdown from the key and list of values from 
def group_from_json(key, values):
    return html.Div([
                html.Label(f"{key}:"),
                dcc.Dropdown(id={'type': 'input-field', 'key': key}, 
                             options=[{'label': option, 'value': option} for option in values], multi=True, value=values[0]),
                    ], className='form-group')

divs = [group_from_json(key, values) for key, values in form_dict.items()]
form = html.Div(divs, className='form-container')


# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# default params
OPAC_DEF = 0.8

# Color mapping
CMAP = "tab20"

# Cross size
CROSS_SIZE = 25


# Create a scatter plot given covariates X (n x 3) and labels y (n x 1)
def create_figure(X, y, titles, id_to_rol_dict, p_x, probs_y):

    # Array containing all the clusters and the user point
    layers = []

    # unique_labels = np.unique(y)
    unique_labels = id_to_rol_dict.keys()

    k = len(unique_labels)

    # create an opacity mapping where higher predicted probability = higher opacity
    if probs_y is not None:
        preds_y = np.argsort(probs_y[0])[::-1]
        p_y = preds_y[0]
        vals = np.linspace(1, 0, k)
        opac_map = dict(zip(preds_y, vals))
    else:
        p_y = None


    # Using colormap from Matplotlib
    color_palette = plt.get_cmap(CMAP)(np.linspace(0, 1, k)) 
    colors = [f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, {c[3]})' for c in color_palette]


    # Do not plot the user point on initialization
    if p_x is not None:

        user_point = go.Scatter3d(
                    x=[p_x[0]], y=[p_x[1]], z=[p_x[2]], 
                    mode='markers', 
                    marker=dict(
                        size=CROSS_SIZE,
                        color=colors[p_y],
                        symbol='cross'
                        ),
                    name=id_to_rol_dict[p_y],
                    text="Predicted Category:",
                    hovertemplate='<b>%{fullData.text}</b><br>%{fullData.name}<extra></extra>',
                    showlegend=False)
        
        layers.append(user_point)

    # Create color-grouped scatter plots for each label in y
    for label in unique_labels:

        # select rows with corresponding label
        X_k = X[y==label]

        # titles should be input to function
        titles_k = titles[y==label]

        x_c, y_c, z_c  = X_k[:,0], X_k[:,1], X_k[:,2] 


        # default or mapped opacity
        if p_y is not None:
            opac = opac_map[label]
            if probs_y[0][label] == 0:
                prob = ""
            else:
                prob = f': {100 * probs_y[0][label]:.3f}%'
        else:
            opac = OPAC_DEF
            prob = ""

        c_val = colors[label]

        # Scatter plot for points
        scatter = go.Scatter3d(
            x=x_c, y=y_c, z=z_c, 
            mode='markers', 
            marker=dict(size=4, color=c_val, opacity=opac),
            name=f"{id_to_rol_dict[label]} {prob}",
            text=titles_k,
            hovertemplate='<b>%{fullData.name}</b><br>%{text}<extra></extra>',
            legendgroup=id_to_rol_dict[label],
            showlegend=False,
        )
        layers.append(scatter)

    # Create figure
    fig = go.Figure(data=layers)
    fig.update_layout(paper_bgcolor='white',
                      scene=dict(aspectmode='data',
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                zaxis=dict(visible=False)),
                      title="Job Sphere",
                      height=1000
                    )

    return fig

# Layout of the app
app.layout = html.Div([
    html.Div([form,
    html.Button('Submit', id='submit-button', n_clicks=0)], style={'width': '25%', 'float': 'left', 'padding': '10px'}),
    html.Div([dcc.Graph(id='3d-scatter', figure=create_figure(X, labels, 
                                                              titles, id_to_rol_dict, 
                                                              None, None), 
                                                              responsive=True)],
             style={'height': '1000 px', 'width': '70%', 'float': 'right'})
])

@app.callback(
    Output('3d-scatter', 'figure'),
    Input('submit-button', 'n_clicks'),
    # capture the input key-value pairs
    State({'type': 'input-field', 'key': ALL}, 'value'),
    State({'type': 'input-field', 'key': ALL}, 'id'),
    State('3d-scatter', 'figure'),
    prevent_initial_call=True,
)
def update_output(n_clicks, input_values, input_ids, prev_fig):
    if n_clicks > 0:
        if None not in input_values:
            # Create a dictionary to hold form data based on the pattern-matched input fields
            form_data = {input_id['key']: value for input_id, value in zip(input_ids, input_values)}

            # Convert to JSON format
            json_output = json.dumps(form_data, indent=4)
            
            # generate template using json_output
            template = concat_output(form_data)
            template_embedding = model.embedder.encode([template])
            X, user_x = model.embedder.get_corrdinates(template_embedding) 

            probs_y = model.predict_proba(template_embedding)
            preds_y = np.argsort(probs_y[0])[::-1]
            user_y = preds_y[0]
            print(f" encoding is: {user_y}")
            # use template to generate embeddings (user_x) and predict probabilities for each label (pred_y))
            # also generate X with reduced dimensions given the user input

            fig = create_figure(X, labels, titles, id_to_rol_dict, user_x, probs_y)
        else:
            fig = prev_fig

        return fig



# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
