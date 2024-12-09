# Yelp Data Analysis and Visualization App

This project is a web-based application built with **Dash** to analyze and visualize Yelp business and review data. The app integrates **AWS Athena** for querying large datasets stored in **S3** and provides interactive data visualizations using **Plotly**.

## Features

- **Integration with AWS Athena**: Query large-scale Yelp datasets efficiently.
- **Interactive Dash Visualizations**:
  - Distribution of star ratings across businesses.
  - Analysis of popular business categories.
  - Visualization of the most reviewed cities.
  - User review distribution insights.
- **Geospatial Visualization**: (Optional) Explore data points on an interactive world map.

## Project Structure

root/ 
├── app.py # Main application logic 
├── requirements.txt
└── README.md # Project documentation


## Prerequisites

To run this application, you need the following:

1. **AWS Credentials**: Configured with permissions to query Athena and access S3.
2. **Athena Database and Tables**: Ensure `yelp-db` is set up with `yelp_academic_dataset_business` and `yelp_academic_dataset_review` tables.
3. **Python Environment**: Python 3.8 or later with required libraries (see `requirements.txt`).

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/yelp-analysis-app.git
   cd yelp-analysis-app
2. ```bash
   pip install -r requirements.txt
3. ```bash
   AWS_ACCESS_KEY_ID=<your-access-key>
   AWS_SECRET_ACCESS_KEY=<your-secret-key>
   REGION_NAME=us-west-1
   S3_BUCKET_NAME=cityu-cs519-tp
4. ```bash
   python app.py


### Key Visualizations
Star Rating Distribution: Displays the distribution of star ratings across businesses.

Popular Business Categories: Highlights the top 20 business categories by frequency.

Most Reviewed Cities: Bar chart showing cities with the most reviews.

User Review Insights:

KDE plot of user review activity.
Cumulative distribution of user reviews.
