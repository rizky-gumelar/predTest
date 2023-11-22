import streamlit as st
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, TheilSenRegressor, HuberRegressor, PassiveAggressiveRegressor, ARDRegression, BayesianRidge, ElasticNet, OrthogonalMatchingPursuit
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

#JUDUL
st.title("Aplikasi Prediksi Dataset")

# UPLOAD DATASET
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    # Can be used wherever a "file-like" object is accepted:
    file_type = uploaded_file.type
    if file_type == "text/csv":
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    st.write(data.head(10))
    # st.write(file_type)

    st.header("Data Column")
    st.write(data.columns)

    st.write(data.isnull().any())
    
    options = st.multiselect(
    'Pilih kolom fitur (X)',data.columns)

    st.write('You selected:', options)

