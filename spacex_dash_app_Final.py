# (download dataset) wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"

# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                dcc.Dropdown(
                                    id='site-dropdown',
                                    options=[
                                        {'label': 'All Sites', 'value': 'ALL'},
                                        {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                                        {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                                        {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                                        {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'}
                                    ],
                                    value='ALL',
                                    placeholder="Select a Launch Site here",
                                    searchable=True
                                ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(
                                    id='payload-slider',
                                    min=min_payload,  # You should define min_payload and max_payload earlier in your code
                                    max=max_payload,
                                    step=1000,
                                    marks={1000: '1000', 5000: '5000', 10000: '10000'},  # Define appropriate marks
                                    value=[min_payload, max_payload]  # Set initial range
                                ),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    if entered_site == 'ALL':
        # Filter data for all sites and calculate successful launches
        success_count = spacex_df[spacex_df['class'] == 1]['class'].count()
        failure_count = spacex_df[spacex_df['class'] == 0]['class'].count()
        fig = px.pie(
            values=[success_count, failure_count],
            names=['Successful', 'Failed'],
            title='Total Successful Launches for All Sites'
        )
    else:
        # Filter data for the selected site and calculate successful launches
        selected_site_data = spacex_df[spacex_df['Launch Site'] == entered_site]
        success_count = selected_site_data['class'].sum()
        failure_count = len(selected_site_data) - success_count
        fig = px.pie(
            values=[success_count, failure_count],
            names=['Successful', 'Failed'],
            title=f'Total Successful Launches for {entered_site}'
        )
    
    return fig
# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    Input(component_id='payload-slider', component_property='value')
)
def update_scatter_chart(payload_range):
    # Filter data based on the payload range
    filtered_df = spacex_df[
        (spacex_df['Payload Mass (kg)'] >= payload_range[0]) &
        (spacex_df['Payload Mass (kg)'] <= payload_range[1])
    ]
    
    # Create a scatter chart to show the correlation between payload and launch success
    fig = px.scatter(
        filtered_df,
        x='Payload Mass (kg)',
        y='class',  # Assuming you have a 'class' column for success
        color='Booster Version Category',  # You can change this to the appropriate column
        title='Correlation between Payload Mass and Launch Success',
        labels={'Payload Mass (kg)': 'Payload Mass (kg)', 'class': 'Launch Status'}
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server()
