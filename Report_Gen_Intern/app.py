import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import plotly.express as px
import re
import plotly.graph_objs as go
import plotly.io as pio
from fpdf import FPDF
from io import BytesIO
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

data = None
cleaned_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        data = pd.read_csv(file)
        columns = data.columns.tolist()
        column_types = data.dtypes.astype(str).to_dict()
        column_data = {col: data[col].dropna().unique().tolist()[:10] for col in columns}  # Limit to first 10 unique values
        raw_data_html = data.head(50).to_html(classes='table table-bordered table-striped', index=False)
        return jsonify({
            'columns': columns,
            'column_types': column_types,
            'column_data': column_data,
            'raw_data': raw_data_html
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/clean_data', methods=['POST'])
def clean_data():
    global data, cleaned_data
    if data is None:
        return jsonify({'error': 'No data available for cleaning.'}), 400

    options = request.json
    cleaned_data = data.copy()

    if options.get('remove_duplicates'):
        cleaned_data = cleaned_data.drop_duplicates()

    if options.get('drop_na'):
        cleaned_data = cleaned_data.dropna()

    cleaned_data_html = cleaned_data.head(50).to_html(classes='table table-bordered table-striped', index=False)
    
    return jsonify({'cleaned_data': cleaned_data_html})

def preprocess_data(df):
    df_processed = df.copy()
    
    # Attempt to identify and convert numeric columns
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            try:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            except:
                pass  # If conversion fails, leave as is

    # Attempt to convert date columns
    date_columns = df_processed.select_dtypes(include=['object']).columns
    for col in date_columns:
        try:
            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
        except:
            pass  # If conversion fails, leave as is

    return df_processed

@app.route('/generate_report', methods=['POST'])
def generate_report():
    global data
    if data is None:
        return jsonify({'error': 'No data available. Please upload a file first.'}), 400
    
    selected_values = request.json
    filtered_data = data.copy()

    for column, value in selected_values.items():
        if value and value != "All":
            filtered_data = filtered_data[filtered_data[column] == value]

    charts = generate_charts(filtered_data)
    return jsonify({'charts': charts})

def generate_charts(data):
    charts = []
    if not data.empty:
        numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()

        # Bar chart
        if numerical_cols and categorical_cols:
            fig = px.bar(data, x=categorical_cols[0], y=numerical_cols[0],
                         title=f'{numerical_cols[0]} by {categorical_cols[0]}')
            charts.append(fig.to_json())

        # Line chart
        if datetime_cols and numerical_cols:
            fig = px.line(data, x=datetime_cols[0], y=numerical_cols[0],
                          title=f'{numerical_cols[0]} Trend')
            charts.append(fig.to_json())

        # Scatter plot
        if len(numerical_cols) > 2:
            fig = px.scatter(data, x=numerical_cols[0], y=numerical_cols[1],
                             color=categorical_cols[0] if categorical_cols else None,
                             title=f'{numerical_cols[0]} vs {numerical_cols[1]}')
            charts.append(fig.to_json())

        # Pie chart
        if categorical_cols:
            pie_data = data[categorical_cols[0]].value_counts()
            fig = px.pie(values=pie_data.values, names=pie_data.index,
                         title=f'Distribution of {categorical_cols[0]}')
            charts.append(fig.to_json())

        # Histogram
        if numerical_cols:
            fig = px.histogram(data, x=numerical_cols[0],
                               title=f'Distribution of {numerical_cols[0]}')
            charts.append(fig.to_json())

        # Box plot
        if numerical_cols and categorical_cols:
            fig = px.box(data, x=categorical_cols[0], y=numerical_cols[0],
                         title=f'{numerical_cols[0]} Distribution by {categorical_cols[0]}')
            charts.append(fig.to_json())

        # Heatmap
        if len(numerical_cols) > 2:
            corr_matrix = data[numerical_cols].corr()
            fig = px.imshow(corr_matrix, title='Correlation Heatmap')
            charts.append(fig.to_json())

        # Treemap
        if len(categorical_cols) > 1 and numerical_cols:
            fig = px.treemap(data, path=categorical_cols[:2], values=numerical_cols[0],
                             title=f'Treemap of {numerical_cols[0]} by {" and ".join(categorical_cols[:2])}')
            charts.append(fig.to_json())

    return charts

@app.route('/analyze', methods=['POST'])
def analyze_data():
    global data
    if data is None or data.empty:
        return jsonify({'error': 'No data available for analysis. Please upload a file first.'}), 400

    analysis_type = request.json.get('analysis_type')
    column1 = request.json.get('column1')
    column2 = request.json.get('column2', None)

    try:
        if analysis_type == 'summary':
            summary = data[column1].describe().to_dict()
            return jsonify({'summary': summary})
        
        elif analysis_type == 'correlation':
            if column2 is None:
                return jsonify({'error': 'Second column is required for correlation analysis.'}), 400
            correlation = data[column1].corr(data[column2])
            return jsonify({'correlation': correlation})
        
        elif analysis_type == 'ttest':
            if column2 is None:
                return jsonify({'error': 'Second column is required for t-test.'}), 400
            t_stat, p_value = stats.ttest_ind(data[column1].dropna(), data[column2].dropna())
            return jsonify({'t_statistic': t_stat, 'p_value': p_value})
        
        elif analysis_type == 'regression':
            if column2 is None:
                return jsonify({'error': 'Second column is required for regression analysis.'}), 400
            X = data[column1].values.reshape(-1, 1)
            y = data[column2].values
            model = LinearRegression().fit(X, y)
            r_squared = model.score(X, y)
            coefficient = model.coef_[0]
            intercept = model.intercept_
            return jsonify({
                'r_squared': r_squared,
                'coefficient': coefficient,
                'intercept': intercept
            })
        
        elif analysis_type == 'clustering':
            if column2 is None:
                return jsonify({'error': 'Second column is required for clustering.'}), 400
            X = data[[column1, column2]].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            return jsonify({
                'clusters': clusters.tolist(),
                'centroids': centroids.tolist()
            })
        
        elif analysis_type == 'time_series':
            if data[column1].dtype != 'datetime64[ns]':
                data[column1] = pd.to_datetime(data[column1])
            time_series = data.set_index(column1)[column2].resample('D').mean()
            return jsonify({
                'dates': time_series.index.strftime('%Y-%m-%d').tolist(),
                'values': time_series.values.tolist()
            })
        
        else:
            return jsonify({'error': 'Invalid analysis type.'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Error during analysis: {str(e)}'}), 400

@app.route('/download_report', methods=['GET'])
def download_report():
    global data, cleaned_data
    if data is None or data.empty:
        return jsonify({'error': 'No data available for the report.'}), 400

    # Preprocess the data
    processed_data = preprocess_data(cleaned_data if cleaned_data is not None else data)

    # Create a PDF report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Data Visualization Report", ln=True, align='C')
    pdf.ln(10)

    # Add charts to the report
    charts = generate_charts(data)
    for i, chart_json in enumerate(charts):
        try:
            fig = pio.from_json(chart_json)
            img_bytes = fig.to_image(format="png")

            temp_img_path = f'temp_chart_{i}.png'
            with open(temp_img_path, 'wb') as img_file:
                img_file.write(img_bytes)

            # Check if the image file exists before trying to use it
            if os.path.exists(temp_img_path):
                pdf.image(temp_img_path, x=10, w=190)
                pdf.ln(5)

                # Remove the temporary image file after using it
                os.remove(temp_img_path)
            else:
                print(f"Failed to generate image: {temp_img_path}")

        except Exception as e:
            print(f"Error generating chart {i}: {e}")
            continue  # Skip this chart and proceed with the next one

    # Add processed data to the report, if available
    if processed_data is not None and not processed_data.empty:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Processed Data", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", '', 8)  # Smaller font size for more columns
        col_widths = [pdf.w / len(processed_data.columns) - 1] * len(processed_data.columns)

        # Add table headers
        for header in processed_data.columns:
            pdf.cell(col_widths[processed_data.columns.get_loc(header)], 7, str(header)[:10], border=1)
        pdf.ln()

        # Add table data (first 50 rows)
        for _, row in processed_data.head(50).iterrows():
            for col, item in row.items():
                pdf.cell(col_widths[processed_data.columns.get_loc(col)], 6, str(item)[:10], border=1)
            pdf.ln()

    # Save the PDF to a temporary file
    temp_pdf_path = 'temp_report.pdf'
    pdf.output(temp_pdf_path)

    # Read the PDF content
    with open(temp_pdf_path, 'rb') as f:
        pdf_output = f.read()

    # Remove the temporary PDF file
    os.remove(temp_pdf_path)

    # Send the PDF file as a response
    return send_file(
        BytesIO(pdf_output),
        as_attachment=True,
        download_name='report.pdf',
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    app.run(debug=True)