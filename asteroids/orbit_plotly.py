import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from astroquery.jplhorizons import Horizons
import dash
from dash.exceptions import PreventUpdate

# 参考: https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=2025%20BG&view=VOP

# "2024 YR4": 衝突するかも
# "2015 BZ509": 木星と逆行する

# 定数
START_DATE = '2030-01-01'
STOP_DATE = '2032-12-31'
ASTEROID_ID = "2024 YR4"

# 惑星データ
PLANETS = {
    '199': 'Mercury', '299': 'Venus', '399': 'Earth', '499': 'Mars',
    '599': 'Jupiter', '699': 'Saturn', '799': 'Uranus', '899': 'Neptune'
}

PLANET_COLORS = {
    'Mercury': '#ff33e6', 'Venus': '#a827f9', 'Earth': '#397df9',
    'Mars': 'red', 'Jupiter': '#f99a39', 'Saturn': '#ffec4a',
    'Uranus': '#4affab', 'Neptune': '#4a8aff'
}

# データ取得
def get_vectors(obj_id):
    return Horizons(id=obj_id, location='500@sun', epochs={'start': START_DATE, 'stop': STOP_DATE, 'step': '1d'}).vectors().to_pandas()

# データ取得
asteroid_df = get_vectors(ASTEROID_ID)
planet_dfs = {pid: get_vectors(pid) for pid in PLANETS}

app = Dash(__name__)

# スライダーラベル作成
marks = {i: asteroid_df.iloc[i]['datetime_str'][:10] for i in range(0, len(asteroid_df), 30)}

# デフォルトカメラ設定
DEFAULT_CAMERA = dict(
    eye=dict(x=2.5, y=2.5, z=1.5),
    up=dict(x=0, y=0, z=1)
)

# アプリのレイアウト設定
app.layout = html.Div([
    dcc.Graph(id='orbit-plot', config={'displayModeBar': False, 'scrollZoom': True}),
    dcc.Slider(
        id='date-slider',
        min=0,
        max=len(asteroid_df) - 1,
        value=0,
        marks=marks,
        step=1
    ),
    html.Div(id='date-display', style={'margin-top': '20px'}),
    html.Button("Prev", id="prev-button", n_clicks=0, style={'font-size': '20px', 'padding': '10px 20px'}),
    html.Button("Play", id="play-button", n_clicks=0, style={'font-size': '20px', 'padding': '10px 20px'}),
    html.Button("Next", id="next-button", n_clicks=0, style={'font-size': '20px', 'padding': '10px 20px'}),
    dcc.Interval(id='interval', interval=500, n_intervals=0, disabled=True)  # 自動再生用のインターバル
])

# グラフ更新のコールバック
@app.callback(
    Output('orbit-plot', 'figure'),
    Input('date-slider', 'value'),
    State('orbit-plot', 'relayoutData')
)
def update_plot(index, relayoutData):
    fig = go.Figure()

    # 小惑星の軌道
    fig.add_trace(go.Scatter3d(
        x=asteroid_df['x'], y=asteroid_df['y'], z=asteroid_df['z'],
        mode='lines', line=dict(color='white', width=2), name=f'{ASTEROID_ID} Orbit'
    ))

    # 現在の小惑星位置
    fig.add_trace(go.Scatter3d(
        x=[asteroid_df.iloc[index]['x']], y=[asteroid_df.iloc[index]['y']], z=[asteroid_df.iloc[index]['z']],
        mode='markers', marker=dict(color='white', size=5), name=ASTEROID_ID
    ))

    # 太陽
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(color='yellow', size=10), name='Sun'
    ))

    # 惑星の軌道と現在位置
    for pid, name in PLANETS.items():
        df = planet_dfs[pid]
        fig.add_trace(go.Scatter3d(
            x=df['x'], y=df['y'], z=df['z'],
            mode='lines', line=dict(color=PLANET_COLORS[name], width=1), name=f'Orbit {name}'
        ))
        fig.add_trace(go.Scatter3d(
            x=[df.iloc[index]['x']], y=[df.iloc[index]['y']], z=[df.iloc[index]['z']], 
            mode='markers', marker=dict(color=PLANET_COLORS[name], size=4), name=name
        ))

    # カメラ設定を取得し、維持する
    if relayoutData and 'scene.camera' in relayoutData:
        camera = relayoutData['scene.camera']
    else:
        camera = DEFAULT_CAMERA

    # レイアウト更新
    fig.update_layout(
        height=600,
        width=1200,
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)',
            xaxis=dict(range=[-15, 15]),
            yaxis=dict(range=[-15, 15]),
            zaxis=dict(range=[-15, 15]),
            aspectmode='cube',
            camera=camera
        ),
        template='plotly_dark'
    )

    return fig

@app.callback(
    Output('date-slider', 'value'),
    Output('date-display', 'children'),
    [Input('next-button', 'n_clicks'), Input('prev-button', 'n_clicks'), Input('interval', 'n_intervals')],
    State('date-slider', 'value')
)
def update_date(next_clicks, prev_clicks, n_intervals, current_value):
    ctx = dash.callback_context

    if not ctx.triggered:
        return current_value, f"Current Date: {asteroid_df.iloc[current_value]['datetime_str']}"

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'next-button' or button_id == 'interval':
        new_value = min(current_value + 1, len(asteroid_df) - 1)
    elif button_id == 'prev-button':
        new_value = max(current_value - 1, 0)
    else:
        new_value = current_value

    return new_value, f"Current Date: {asteroid_df.iloc[new_value]['datetime_str']}"

@app.callback(
    Output('interval', 'disabled'),
    Input('play-button', 'n_clicks'),
    State('interval', 'disabled')
)
def toggle_play(play_clicks, interval_disabled):
    if play_clicks is None:
        raise PreventUpdate

    return not interval_disabled

if __name__ == '__main__':
    app.run_server(debug=True)
