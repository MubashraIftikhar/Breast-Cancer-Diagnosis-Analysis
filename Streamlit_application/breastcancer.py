import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Using Logistic Regression
from sklearn.metrics import classification_report, confusion_matrix
from GraphicGenerator import GraphicGenerator
import base64
from PIL import Image

# Load the dataset
def load_data():
    try:
        data = pd.read_csv('data.csv')  # Replace with your actual file path
        return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'data.csv' is in the correct path.")
        return None

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def add_header_image(local_image_path, text_color="white"):
    bin_str = get_base64_of_bin_file(local_image_path)
    css = f"""
    <style>
    .header-container {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        border-radius: 5px;
        padding: 80px;
    }}
    .header-text {{
        position: absolute;  /* Position the text absolutely */
        top: 20px;  /* Distance from the top */
        left: 20px;  /* Distance from the left */
        color: {text_color};  /* Dynamic text color */
        font-size: 35px;  /* Font size */
        font-weight: bold;  /* Font weight */
        padding: 5px;  /* Padding around the text */
        border-radius: 5px;  /* Rounded corners */
    }}
    .subheader-text {{ 
        padding-left: 40px; 
        padding-top: 10px;
        font-size: 18px; 
        font-weight: normal;
        color: {text_color};  /* Dynamic text color */
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def introduction():
    add_header_image("C:/Users/Lenovo/Desktop/DS/breast_cancer/bc3.jpeg","#e2619f")  # Replace with your local file path

    st.markdown("""
    <div class="header-container">
        <div class="header-text">Breast Cancer Predictor</div>
        <div class="subheader-text">Predict whether the cancer is benign or malignant</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
    This application predicts whether a tumor is malignant or benign based on various features from the Breast Cancer Wisconsin dataset.
    The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.""")

    st.write("### Objective")
    st.write("""
    The **objective** of this project is to conduct exploratory data analysis (EDA) to study patterns, correlations, and distributions within the dataset. 
    The ultimate goal of this project is to develop a robust machine learning model capable of accurately classifying tumors as malignant or benign. 
    A key focus will be on identifying the most significant features that contribute to tumor classification, thereby enhancing the model's performance and predictive accuracy.
    """)

    st.write("### Characteristics of Breast Cancer Wisconsin (Diagnostic) Dataset")

    # Styled table for Characteristics
    html_table_characteristics = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th {
            background-color: #ff69b4;  /* Pink background */
            color: white;              /* White text */
            text-align: left;
            padding: 10px;
            font-size: 16px;
        }
        td {
            text-align: left;
            padding: 10px;
            font-size: 14px;
            border-bottom: 1px solid #ddd; /* Light gray border for rows */
        }
        tr:nth-child(even) {
            background-color: #f9f9f9; /* Light gray background for even rows */
        }
        tr:hover {
            background-color: #f1f1f1; /* Slightly darker gray for hover effect */
        }
    </style>
    <table>
        <thead>
            <tr>
                <th>Characteristic</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Number of Instances</td>
                <td>569</td>
            </tr>
            <tr>
                <td>Number of Attributes</td>
                <td>30 numerical attributes used for prediction, along with a class label</td>
            </tr>
            <tr>
                <td>Class Distribution</td>
                <td>212 - Malignant, 357 - Benign</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(html_table_characteristics, unsafe_allow_html=True)

    st.write("### Attributes of Breast Cancer Wisconsin (Diagnostic) Dataset")

    # Styled table for Features
    html_table_features = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th {
            background-color: #ff69b4;  /* Pink background */
            color: white;              /* White text */
            text-align: left;
            padding: 10px;
            font-size: 16px;
        }
        td {
            text-align: left;
            padding: 10px;
            font-size: 14px;
            border-bottom: 1px solid #ddd; /* Light gray border for rows */
        }
        tr:nth-child(even) {
            background-color: #f9f9f9; /* Light gray background for even rows */
        }
        tr:hover {
            background-color: #f1f1f1; /* Slightly darker gray for hover effect */
        }
    </style>
    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>mean radius</td>
                <td>Mean of distances from center to points on the perimeter.</td>
            </tr>
            <tr>
                <td>mean texture</td>
                <td>Standard deviation of gray-scale values.</td>
            </tr>
            <tr>
                <td>mean perimeter</td>
                <td>Perimeter of the tumor.</td>
            </tr>
            <tr>
                <td>mean area</td>
                <td>Area of the tumor.</td>
            </tr>
            <tr>
                <td>mean smoothness</td>
                <td>Variation in radius lengths.</td>
            </tr>
            <tr>
                <td>mean compactness</td>
                <td>Perimeter² / Area - 1.0.</td>
            </tr>
            <tr>
                <td>mean concavity</td>
                <td>Severity of concave portions of the contour.</td>
            </tr>
            <tr>
                <td>mean concave points</td>
                <td>Number of concave portions of the contour.</td>
            </tr>
            <tr>
                <td>mean symmetry</td>
                <td>Symmetry of the cell nuclei.</td>
            </tr>
            <tr>
                <td>mean fractal dimension</td>
                <td>"Coastline approximation" - 1.</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(html_table_features, unsafe_allow_html=True)



import base64

def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        # Use base64.b64encode for encoding and decode to get a string
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_image});
            background-attachment: fixed;
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def eda(data):
    # Add background image
    add_bg_from_local("C:/Users/Lenovo/Desktop/DS/breast_cancer/eda2.png")  # Replace with your local file path
    
    add_header_image("C:/Users/Lenovo/Desktop/DS/breast_cancer/model.jpg")  # Replace with your local file path

    st.markdown("""
    <div class="header-container">
        <div class="header-text">Exploratory Data Analysis (EDA)</div>
        <div class="subheader-text">Here are some insights from the dataset</div>
    </div>
    """, unsafe_allow_html=True)
  

    # Drop the 'id' column if present
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
    if 'Unnamed: 32' in data.columns:
        data = data.drop(columns=['Unnamed: 32'])

    # Custom CSS for data frame headers
    st.markdown(
        """
        <style>
        .dataframe thead th {
            background-color: pink;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the dataset
    st.subheader("Dataset Preview")
    st.dataframe(data.head())
    
    # Display basic statistics
    st.subheader("Dataset Summary Statistics")
    st.write(data.describe())
    
    # Distribution of Tumor Types
    st.subheader("Distribution of Tumor Types")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='diagnosis', data=data, palette='Blues')
    plt.title("Tumor Type Distribution")
    st.pyplot(plt)
    plt.clf()  # Clear the figure

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    matrix = data.corr()  # Compute correlation matrix
    plt.figure(figsize=(20, 12))
    mask = np.triu(np.ones_like(matrix, dtype=bool))  # Create the mask for the upper triangle
    sns.heatmap(matrix, mask=mask, linewidths=1, annot=True, fmt=".2f", cmap='Blues')
    st.pyplot(plt)
    plt.clf()  # Clear the figure

    st.header("Graphic Plots")
    checked_pairplot = st.checkbox('PairPlot')
    checked_scatterPlot = st.checkbox('ScatterPlot')
    

    plotGenerator = GraphicGenerator(data)

    if checked_pairplot:
        plotGenerator.pairplot()
        st.markdown('<hr/>', unsafe_allow_html=True)

    if checked_scatterPlot:
        plotGenerator.scatterplot()
        st.markdown('<hr/>', unsafe_allow_html=True)

def add_custom_css():
    st.markdown(
        """
        <style>
        /* Change the background color of multiselect input tags */
        div[data-baseweb="select"] > div {
            background-color: #cce7ff !important; /* Light blue background */
            border-radius: 5px !important; /* Rounded corners */
        }

        /* Change the background of selected options in the multiselect */
        div[data-baseweb="tag"] {
            background-color: #0073e6 !important; /* Blue background */
            color: white !important; /* White text */
            border-radius: 5px !important; /* Rounded corners */
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def model(data):
    add_bg_from_local("C:/Users/Lenovo/Desktop/DS/breast_cancer/eda.png") 
    # Drop unnecessary columns
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
    if 'Unnamed: 32' in data.columns:
        data = data.drop(columns=['Unnamed: 32'])
    add_header_image("C:/Users/Lenovo/Desktop/DS/breast_cancer/model2.png")  # Replace with your local file path

    st.markdown("""
    <div class="header-container">
        <div class="header-text">Model Training & Predictions</div>
    </div>
    """, unsafe_allow_html=True)
   
    # Columns for multiselect (excluding the target column 'diagnosis')
    _self_columns = list(data.columns)
    _self_binary_columns = ['diagnosis']  # Assuming 'diagnosis' is the only binary column


    X_columns = st.multiselect(
        'Select Features (X)',
        _self_columns,
        default=[col for col in _self_columns if col != 'diagnosis'],
        key='x_logistic_reg'
    )

    y_column = st.selectbox(
        'Select Target Variable (y)',
        _self_binary_columns,
        key='y_logistic_reg'
    )

    if not X_columns or not y_column:
        st.error("Please select at least one feature and a target variable to proceed.")
        return

    X = data[X_columns]
    y = data[y_column].map({'M': 1, 'B': 0})  # Assuming binary classification (Malignant: 1, Benign: 0)

    # PART 2: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # PART 3: Allow testing with user-selected input
    st.subheader("Test the Model with Custom Input")
    st.write("Provide values for the selected features:")
    
    user_input = []
    if X_columns:  # Check if features are selected
        for feature in X_columns:
            value = st.slider(
                f"Select value for {feature}",
                min_value=float(X[feature].min()),
                max_value=float(X[feature].max()),
                value=float(X[feature].mean())
            )
            user_input.append(value)
    else:
        st.warning("No features selected for input range.")

    if st.button("Predict"):
        if len(user_input) == len(X_columns):  # Ensure input matches the selected features
            prediction = model.predict([user_input])
            result = "Malignant (1)" if prediction[0] == 1 else "Benign (0)"
            st.success(f"The predicted diagnosis is: {result}")
        else:
            st.error("Please provide values for all selected features.")

    # PART 4: Display model performance at the end
    add_header_image("C:/Users/Lenovo/Desktop/DS/breast_cancer/model2.png")  # Replace with your local file path

    st.markdown("""
    <div class="header-container">
        <div class="header-text">Model Evaluation</div>
    </div>
    """, unsafe_allow_html=True)
    # Predictions and metrics
    y_pred = model.predict(X_test)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign (0)', 'Malignant (1)'],
                yticklabels=['Benign (0)', 'Malignant (1)'])
    plt.title("Confusion Matrix")
    st.pyplot(plt)
    plt.clf()

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)


# Conclusion Section
def conclusion():
    
    add_bg_from_local("C:/Users/Lenovo/Desktop/DS/breast_cancer/eda2.png")  # Replace with your local file path
    
    add_header_image("C:/Users/Lenovo/Desktop/DS/breast_cancer/conc.png")  # Replace with your local file path

    st.markdown("""
    <div class="header-container">
        <div class="header-text">Conclusion</div>
    </div>
    """, unsafe_allow_html=True)
  
    st.write("""
    The model successfully predicts whether a tumor is malignant or benign with a high degree of accuracy.
    The exploratory data analysis provided insights into the distribution of tumor types and the relationships between features.
    The Model Pane allows users to select features (X) and the target variable (y) for training a Logistic Regression model.
              It provides tools to evaluate model performance, including a classification report and a confusion matrix.
              Additionally, users can test the model by inputting custom feature values to predict outcomes (e.g., benign or malignant diagnosis).
    """)

def apply_custom_styles():
    # Custom CSS for background color
    st.markdown("""
    <style>
       
        .main > div {
            background-color: #ffffff ;
            border-radius: 12px;   
            padding: 25px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);  /* Subtle shadow for a raised effect */
        }

    </style>
    """, unsafe_allow_html=True)

# Main function to run the app
def main():
    data = load_data()
    if data is not None:
        # Sidebar Navigation
        st.sidebar.title("Navigation")
        options = ["Introduction", "EDA", "Model", "Conclusion"]
        choice = st.sidebar.radio("Go to:", options)

        if choice == "Introduction":
            #apply_custom_styles()
            introduction()
        elif choice == "EDA":
            eda(data)
        elif choice == "Model":
            model(data)
        elif choice == "Conclusion":
            conclusion()
    else:
        st.error("Unable to load the dataset. Please check the file path.")

if __name__ == "__main__":
    main()
