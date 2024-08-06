# Import
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import dash
import boto3
import time
from dash import dcc, html, dash_table
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot #for offline ploting

athena_client = boto3.client('athena', region_name='us-west-1')
s3 = boto3.resource("s3")

def query_athena(query, database, s3_output):
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': s3_output}
    )
    query_execution_id = response['QueryExecutionId']
    
    while True:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = response['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)
    
    if state == 'SUCCEEDED':
        results = athena_client.get_query_results(QueryExecutionId=query_execution_id)
        local_filename = "athena_query_result_temp.csv"
        s3.Bucket('cityu-cs519-tp').download_file("output/athena-query-output/"+query_execution_id + ".csv", local_filename)
        dfe = pd.read_csv(local_filename)
        return dfe
    else:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        error_message = response['QueryExecution']['Status']['StateChangeReason']
        raise Exception(f"Query failed with state: {state}, reason: {error_message}")

def results_to_dataframe(results):
    columns = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
    rows = []
    for row in results['ResultSet']['Rows'][1:]:  # Skip the header row
        rows.append([col.get('VarCharValue', '') for col in row['Data']])
    df = pd.DataFrame(rows, columns=columns)

    return df

def create_map_image(df):
    print("Data before cleaning:")
    print(df)
    
    df = df.dropna(subset=['longitude', 'latitude'])
    df = df[(df['longitude'] != '') & (df['latitude'] != '')]
    
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df = df.dropna(subset=['longitude', 'latitude'])
    
    print("Data after cleaning:")
    print(df)
    
    df['longitude'] = df['longitude'].astype(float)
    df['latitude'] = df['latitude'].astype(float)

    fig, ax = plt.subplots(figsize=(15, 10))
    m = Basemap(
        projection='merc',
        llcrnrlat=df['latitude'].min() - 5, urcrnrlat=df['latitude'].max() + 5,
        llcrnrlon=df['longitude'].min() - 5, urcrnrlon=df['longitude'].max() + 5,
        lat_ts=20, resolution='i', ax=ax
    )
    
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.bluemarble()
    x, y = m(df['longitude'].values, df['latitude'].values)
    m.scatter(x, y, color='red', marker='o', s=50)
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return f"data:image/png;base64,{image_base64}"

def data_restructure_bus(df):
    # Ratings,reviews and locations 
    df['latitude'] = pd.to_numeric(df['latitude'])
    df['longitude'] = pd.to_numeric(df['longitude'])
    df['stars'] = pd.to_numeric(df['stars'])
    df['review_count'] = pd.to_numeric(df['review_count'])
    df['is_open'] = pd.to_numeric(df['is_open'])
    # company_rating=df.groupby(['name'])['stars'].mean()
    # company_review_count=df.groupby(['name']).review_count.sum()
    # company_city_count=df.groupby(['name']).city.nunique()
    # result = pd.concat([company_review_count, company_rating,company_city_count], axis=1, join='inner')
    # result.sort_values(by=['city'], ascending=False).head(10)
    return df

def data_restructure_rev(b_df, r_df):
    print("gkerogepkp")
    new_dataset_good, new_dataset_bad = company_selection_rev(b_df, r_df)
    print(new_dataset_good, new_dataset_bad)
    grouped_df_good = new_dataset_good.groupby(by=["name", "stars_y"]).size().reset_index(name="counts")
    print("Groupedd ===> ", grouped_df_good)
    grouped_df_good['percentage'] = 100 * grouped_df_good['counts'] / grouped_df_good.groupby('name')['counts'].transform('sum')
    plot_1=px.bar(grouped_df_good, 
       x='name', y=['counts'], color='stars_y', 
       text=grouped_df_good['percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
        labels={
                     "name": "Companies",
                     "value": "Reviews Distribution",
                     "stars_y":"Legend"
                 },
                title="Distribution of Reviews Among Companies That Are Doing Good")

    # iplot(plot_1, filename='plot_1')
    return plot_1

def company_selection_rev(businesses, reviews):
    businesses_good=businesses[(businesses['name']=="European Wax Center") |
                (businesses['name']=="Nothing Bundt Cakes") |
               (businesses['name']=="Take 5 Oil Change") |
               (businesses['name']=="Trader Joe's") |
               (businesses['name']=="Discount Tire")]
    biz_ids_good=businesses_good.business_id.unique()
    
#poorly performing businesses 
    businesses_bad=businesses[(businesses['name']=="McDonald's") |
                (businesses['name']=="Starbucks") |
               (businesses['name']=="Domino's Pizza") |
               (businesses['name']=="Burger King") |
               (businesses['name']=="Subway")]
    biz_ids_bad=businesses_bad.business_id.unique()
    print("saijfosj", biz_ids_good, biz_ids_bad)
    reviews['stars'] = reviews['stars'].astype(float)
    reviews['stars'] = reviews['stars'].astype(float)
    yelp_review_good=businesses.loc[reviews['business_id'].isin(biz_ids_good)]
    yelp_review_bad=businesses.loc[reviews['business_id'].isin(biz_ids_bad)]
    # del(yelp_review)
    # del(businesses)
    new_dataset_good=pd.merge(businesses_good,yelp_review_good,on='business_id')
    new_dataset_bad=pd.merge(businesses_bad,yelp_review_bad,on='business_id')
    return new_dataset_good, new_dataset_bad


def ratings_distribution(b_df):
    x = b_df['stars'].value_counts().sort_index()
    fig1 = go.Figure(data=[go.Bar(x=x.index, y=x.values)])
    fig1.update_layout(
        title="Star Rating Distribution",
        xaxis_title="Star Ratings",
        yaxis_title="# of businesses",
        font=dict(size=14),
        bargap=0.2,  # Adjust bar spacing
        margin=dict(l=50, r=50, t=80, b=50)
    )
    for i, label in enumerate(x.values):
        fig1.add_annotation(
        x=x.index[i],
        y=x.values[i] + 5,
        text=str(label),
        showarrow=False,
        font=dict(size=10)
    )
    return fig1

def popular_categories(b_df):
    business_cats = ' '.join(b_df['categories'])
    cats = pd.DataFrame(business_cats.split(','), columns=['category'])
    x = cats.category.value_counts().sort_values(ascending=False).iloc[0:20]

    fig = go.Figure(data=[go.Bar(x=x.index, y=x.values)])

    fig.update_layout(
    title="Top 20 Business Categories",
    xaxis_title="Category",
    yaxis_title="# of Businesses",
    font=dict(size=12),
    bargap=0.2,
    title_font=dict(size=25),
    xaxis_tickfont=dict(size=10),
    yaxis_tickfont=dict(size=10),
    margin=dict(t=80, b=50)
)

# Rotate x-axis labels
    fig.update_xaxes(tickangle=80)

# Add text labels
    for i, label in enumerate(x.values):
        fig.add_annotation(
        x=x.index[i],
        y=x.values[i] + 5,
        text=str(label),
        showarrow=False,
        font=dict(size=10)
    )
        
        return fig

def world_view(b_df):
    # Create the map trace
    data = go.Scattergeo(
    lon=b_df['longitude'],
    lat=b_df['latitude'],
    mode='markers',
    marker=dict(
        size=3,
        color='orange',
        line=dict(
            width=3,
            color='orange'
        ),
        opacity=1
    )
)

# Create the layout
    layout = go.Layout(
    title='World-wide Yelp Reviews',
    geo=dict(
        projection=dict(type='orthographic', rotation=dict(lon=-50, lat=20)),
        showland=True,
        landcolor='#bbdaa4',
        showocean=True,
        oceancolor='#4a80f5',
        showcountries=True,
        countrycolor='black',
        countrywidth=0.1
    )
)

# Combine the data and layout
    fig = go.Figure(data=data, layout=layout)
    return fig

def city_most_reviews(b_df):
    city_counts = b_df['city'].value_counts().sort_values(ascending=False).iloc[:20]
    fig = go.Figure(data=[go.Bar(
        x=city_counts.index,
        y=city_counts.values,
        marker_color='lightblue'  # Choose a color for the bars
    )])
    fig.update_layout(
        title="Which city has the most reviews?",
        xaxis_tickangle=-45,  # Rotate x-axis labels
        yaxis_title="# businesses",
        xaxis_title="City"
    )
    for i, v in enumerate(city_counts.values):
        fig.add_annotation(
        x=city_counts.index[i],
        y=v,
        text=str(v),
        showarrow=False,
        yshift=10
        )
    return fig

def user_ops(rev_df):
    user_agg=rev_df.groupby('user_id', as_index=False).agg({'review_id':['count'],'date':['min','max'],
                                'useful':['sum'],'funny':['sum'],'cool':['sum'],
                               'stars':['mean']})

    # user_agg = [{'user_id': k} | g.drop(columns='user_id')
    #    for k, g in rev_df.groupby('user_id', as_index=False)]
    user_agg=user_agg.sort_values([('review_id','count')],ascending=False)
    return user_agg

def deep_dive(user_agg):
    user_agg[('review_id', 'count')] = user_agg[('review_id', 'count')].apply(lambda x: 30 if x > 30 else x)

    trace1 = go.Scattergl(x=user_agg[('review_id', 'count')],
                    y=user_agg[('review_id', 'count')].sort_values().index,
                    mode='lines+markers',
                    fill='tozeroy',
                    name='KDE Plot')

# Create the cumulative distribution plot
    trace2 = go.Scattergl(x=user_agg[('review_id', 'count')],
                    y=user_agg[('review_id', 'count')].rank(method='dense', ascending=True) / len(user_agg),
                    mode='lines+markers',
                    name='Cumulative Distribution')

    layout = go.Layout(
        title='User Review Distribution',
        xaxis=dict(title='Number of Reviews Given'),
        yaxis=dict(title='Number of Users', anchor='x', range=[0, 1]),
        showlegend=True,
        legend=dict(x=0.8, y=1),
        hovermode='closest'
    )
    fig1  = go.Figure(data=trace1, layout=layout)
    fig2  = go.Figure(data=trace2, layout=layout)
    return fig1, fig2


app = dash.Dash(__name__)

business_query = "SELECT * FROM yelp_academic_dataset_business limit 70000;" #Limit can be removed
review_query = "SELECT * FROM yelp_academic_dataset_review limit 50000;" #Limit can be removed

database = "yelp-db"
s3_output = "s3://cityu-cs519-tp/output/athena-query-output/"

try:
    bus_result = query_athena(business_query, database, s3_output)
    # bus_df = results_to_dataframe(bus_result)
    bus_df = data_restructure_bus(bus_result)
    bus_df = bus_df.dropna()

    fig1 = ratings_distribution(bus_df)
    fig2 = popular_categories(bus_df)
    # fig3 = world_view(bus_df)
    fig4 = city_most_reviews(bus_df)


    # map_image = create_map_image(df)
    rev_results = query_athena(review_query, database, s3_output)
    user_info = user_ops(rev_results)
    kde_plot, dist_plot = deep_dive(user_info)
    # rev_df = results_to_dataframe(rev_results)
    # print("helllo 2 ===>", rev_df)

    # plot_1 = data_restructure_rev(bus_result, rev_df)

    # rev_data = data_restructure_bus(rev_results)
    # bus_df = results_to_dataframe(results)
    app.layout = html.Div(children=[
        # html.Img(src=map_image)
        dcc.Graph(
            id='star-rating-distribution',
            figure=fig1,
            style={'width': '48%', 'display': 'inline-block'}
        ),
        dcc.Graph(
            id='top-business-categories',
            figure=fig2,
            style={'width': '48%', 'display': 'inline-block'}
        ),
        # dcc.Graph(
        #     id='world-view',
        #     figure=fig3
        # ),
        dcc.Graph(
            id='city-review',
            figure=fig4
        ),
        dcc.Graph(id='kde-plot', figure=kde_plot, style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='dist-plot', figure=dist_plot, style={'width': '48%', 'display': 'inline-block'}),

        # dcc.Graph(id='dist-plot', figure=dist_plot,style={'width': '48%', 'display': 'flex'})

    ])
except Exception as e:
    print(e)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')