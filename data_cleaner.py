import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataCleaner:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)

    def show_info(self):
        buffer = []
        self.data.info(buf=buffer)
        return "\n".join(buffer)

    def show_description(self):
        return self.data.describe(include='all').to_string()

    def show_missing(self):
        return self.data.isnull().sum().to_frame(name='Missing Values')

    def remove_duplicates(self):
        self.data.drop_duplicates(inplace=True)

    def handle_missing(self, method='mean'):
        for col in self.data.select_dtypes(include=np.number).columns:
            if method == 'mean':
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            elif method == 'median':
                self.data[col].fillna(self.data[col].median(), inplace=True)
        for col in self.data.select_dtypes(include='object').columns:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)

    def standardize_columns(self):
        self.data.columns = self.data.columns.str.lower().str.replace(' ', '_')

    def encode_categoricals(self):
        le = LabelEncoder()
        for col in self.data.select_dtypes(include='object').columns:
            self.data[col] = le.fit_transform(self.data[col].astype(str))

    def scale_numeric(self):
        scaler = StandardScaler()
        num_cols = self.data.select_dtypes(include=np.number).columns
        self.data[num_cols] = scaler.fit_transform(self.data[num_cols])

    def auto_clean(self):
        self.remove_duplicates()
        self.handle_missing(method='mean')
        self.standardize_columns()
        self.encode_categoricals()
        self.scale_numeric()

    def save(self, output_path='cleaned_output.csv'):
        self.data.to_csv(output_path, index=False)

    def get_data(self):
        return self.data