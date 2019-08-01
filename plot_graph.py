import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
from sklearn.preprocessing import MinMaxScaler as MMS
init_notebook_mode(connected=True)

def plot_graph(graph=nx.tutte_graph(), layout=nx.kamada_kawai_layout):
    """
    Make plotly visualization of networkx graph.
    node_size -> betweeness centrality
    node_color -> closeness centrality
    """
    b_cents = nx.betweenness_centrality(graph)
    c_cents = nx.closeness_centrality(graph)
    d_cents = graph.degree()
    edge_x = []
    edge_y = []
    pos = layout(graph)
    for edge in graph.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    node_x = []
    node_y = []
    node_texts = []
    node_degrees = []
    node_closenesses = []
    node_betweenesses = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_degree = d_cents[node]
        node_closeness = c_cents[node]
        node_betweeness = b_cents[node]
        node_degrees.append(node_degree)
        node_closenesses.append(node_closeness)
        node_betweenesses.append(node_betweeness)
        node_text = str(node) + "<br>Degree: " + str(node_degree)
        node_texts.append(node_text)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            opacity=1,
            showscale=True,
            colorscale='Jet',
            reversescale=False,
            color=[],
            size=[],
            colorbar=dict(
                thickness=15,
                title='Closeness',
                xanchor='left',
                titleside='right'
            ),
            line=dict(
                width=1,
                color='Black'
            ),
        )
    )
    node_degrees = np.array(node_degrees)
    node_closenesses = np.array(node_closenesses)
    node_betweenesses = np.array(node_betweenesses)
    size_scaler = MMS(feature_range=(7.5, 17.5))
    node_betweenesses = size_scaler.fit_transform(node_betweenesses.reshape(-1,1)).ravel()
    node_trace.marker.color = node_closenesses
    node_trace.marker.size = node_betweenesses
    node_trace.text = node_texts
    fig = go.Figure(
        data=[
            edge_trace,
            node_trace
        ],
        layout=go.Layout(
            autosize=True,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )
    iplot(fig)
