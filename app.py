import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="House Price AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏠 House Price AI")

st.write(
    "Machine Learning based House Price Prediction "
    "using Random Forest Regression."
)

st.caption(
    "Enter property details in the sidebar to estimate the house sale price."
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    file_path = "dataset/train.csv"

    try:
        df = pd.read_csv(file_path)

    except Exception:
        df = pd.read_csv(
            file_path,
            engine="python"
        )

    return df


df = load_data()


# ============================================================
# FEATURES
# ============================================================

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "YearBuilt"
]

target = "SalePrice"


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = features + [target]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"These columns are missing from train.csv: "
        f"{missing_columns}"
    )

    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

model_data = df[required_columns].copy()

for column in required_columns:

    model_data[column] = pd.to_numeric(
        model_data[column],
        errors="coerce"
    )

model_data = model_data.dropna()


# ============================================================
# X AND Y
# ============================================================

X = model_data[features]

y = model_data[target]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# TRAIN RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def train_model(X_train_data, y_train_data):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train_data,
        y_train_data
    )

    return model


model = train_model(
    X_train,
    y_train
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# MODEL METRICS
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.subheader("📊 Project Overview")

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "🏠 Total Houses",
        f"{len(model_data):,}"
    )


with metric2:

    average_price = model_data[target].mean()

    st.metric(
        "💰 Average Price",
        f"${average_price:,.0f}"
    )


with metric3:

    st.metric(
        "🤖 Model",
        "Random Forest"
    )


with metric4:

    st.metric(
        "📈 R² Score",
        f"{r2:.3f}"
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏠 House Details")

st.sidebar.write(
    "Enter the property details to predict its price."
)

st.sidebar.divider()


# ============================================================
# OVERALL QUALITY
# ============================================================

overall_qual = st.sidebar.slider(
    "⭐ Overall Quality",
    min_value=1,
    max_value=10,
    value=7
)


# ============================================================
# LIVING AREA
# ============================================================

gr_liv_area = st.sidebar.number_input(
    "📐 Living Area (sq ft)",
    min_value=300,
    max_value=5000,
    value=1800,
    step=50
)


# ============================================================
# GARAGE CARS
# ============================================================

garage_cars = st.sidebar.number_input(
    "🚗 Garage Capacity",
    min_value=0,
    max_value=5,
    value=2,
    step=1
)


# ============================================================
# GARAGE AREA
# ============================================================

garage_area = st.sidebar.number_input(
    "🚗 Garage Area (sq ft)",
    min_value=0,
    max_value=1500,
    value=500,
    step=50
)


# ============================================================
# BASEMENT
# ============================================================

total_bsmt_sf = st.sidebar.number_input(
    "🏠 Basement Area (sq ft)",
    min_value=0,
    max_value=3000,
    value=1000,
    step=50
)


# ============================================================
# FIRST FLOOR
# ============================================================

first_flr_sf = st.sidebar.number_input(
    "📐 First Floor Area (sq ft)",
    min_value=300,
    max_value=3000,
    value=1000,
    step=50
)


# ============================================================
# BATHROOMS
# ============================================================

full_bath = st.sidebar.number_input(
    "🛁 Full Bathrooms",
    min_value=0,
    max_value=5,
    value=2,
    step=1
)


# ============================================================
# YEAR BUILT
# ============================================================

year_built = st.sidebar.number_input(
    "📅 Year Built",
    min_value=1800,
    max_value=2026,
    value=2005,
    step=1
)


st.sidebar.divider()


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "🔮 Predict House Price",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN PREDICTION SECTION
# ============================================================

left, right = st.columns(2)


# ============================================================
# PROPERTY OVERVIEW
# ============================================================

with left:

    st.subheader("🏡 Property Overview")

    st.write(
        "These are the values that will be given "
        "to the Machine Learning model."
    )

    st.write(
        f"⭐ **Overall Quality:** {overall_qual}/10"
    )

    st.progress(
        overall_qual / 10
    )

    st.write(
        f"📐 **Living Area:** {gr_liv_area:,} sq ft"
    )

    st.write(
        f"🚗 **Garage:** {garage_cars} cars"
    )

    st.write(
        f"🚗 **Garage Area:** {garage_area:,} sq ft"
    )

    st.write(
        f"🏠 **Basement:** {total_bsmt_sf:,} sq ft"
    )

    st.write(
        f"📐 **First Floor:** {first_flr_sf:,} sq ft"
    )

    st.write(
        f"🛁 **Full Bathrooms:** {full_bath}"
    )

    st.write(
        f"📅 **Year Built:** {year_built}"
    )


# ============================================================
# PRICE PREDICTION
# ============================================================

with right:

    st.subheader("💰 Price Prediction")

    st.write(
        "Click the button in the sidebar to predict "
        "the estimated house price."
    )

    if predict_button:

        new_house = pd.DataFrame({

            "OverallQual": [
                overall_qual
            ],

            "GrLivArea": [
                gr_liv_area
            ],

            "GarageCars": [
                garage_cars
            ],

            "GarageArea": [
                garage_area
            ],

            "TotalBsmtSF": [
                total_bsmt_sf
            ],

            "1stFlrSF": [
                first_flr_sf
            ],

            "FullBath": [
                full_bath
            ],

            "YearBuilt": [
                year_built
            ]

        })


        prediction = model.predict(
            new_house
        )[0]


        # Save prediction

        history_item = {

            "Overall Quality": overall_qual,

            "Living Area": gr_liv_area,

            "Garage Cars": garage_cars,

            "Garage Area": garage_area,

            "Predicted Price": prediction

        }


        st.session_state.prediction_history.append(
            history_item
        )


        st.success(
            "✅ Prediction generated successfully!"
        )


        st.metric(
            "🏠 Estimated Sale Price",
            f"${prediction:,.0f}"
        )


        st.write(
            f"Approximately **${prediction:,.0f}**"
        )


    else:

        st.info(
            "👈 Enter house details in the sidebar "
            "and click **Predict House Price**."
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.divider()

st.subheader("🕘 Prediction History")

if len(st.session_state.prediction_history) > 0:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    history_df.index = history_df.index + 1

    history_df.index.name = "Prediction #"

    st.dataframe(
        history_df,
        use_container_width=True
    )


    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()

else:

    st.info(
        "No predictions yet. Make your first prediction "
        "using the sidebar."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.subheader("📈 Machine Learning Insights")

st.write(
    "The chart below shows which features have the "
    "largest importance in the Random Forest model."
)


importance_df = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


st.bar_chart(
    importance_df.set_index("Feature")
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🎯 Model Performance")

performance1, performance2 = st.columns(2)


with performance1:

    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )


with performance2:

    st.metric(
        "Mean Absolute Error",
        f"${mae:,.0f}"
    )


st.write(
    "The R² score indicates how well the model explains "
    "variation in house prices on the test dataset."
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.divider()

st.subheader("📊 Actual vs Predicted House Prices")

st.write(
    "This chart compares actual house prices from the "
    "test dataset with prices predicted by the Random Forest model."
)


comparison_df = pd.DataFrame({

    "Actual Price": y_test.values,

    "Predicted Price": y_pred

})


st.scatter_chart(
    comparison_df,
    x="Actual Price",
    y="Predicted Price"
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("📋 Project Information")

info1, info2, info3 = st.columns(3)


with info1:

    st.write("### 🤖 Machine Learning")

    st.write(
        "Random Forest Regression is used to "
        "predict house sale prices."
    )


with info2:

    st.write("### 📊 Dataset")

    st.write(
        "Ames Housing / Kaggle House Prices dataset "
        "containing house property information."
    )


with info3:

    st.write("### 💻 Technologies")

    st.write(
        "Python, Pandas, Scikit-learn and Streamlit."
    )


# ============================================================
# SELECTED FEATURES
# ============================================================

st.divider()

st.subheader("🔎 Features Used by Model")

feature_text = ", ".join(features)

st.write(feature_text)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏠 House Price AI | Machine Learning Portfolio Project"
)