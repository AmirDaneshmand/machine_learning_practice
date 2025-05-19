import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self, df):
        self.df = df.copy()

    def handle_missing_values(self):
        # حذف سفارشات بدون مشتری یا بدون کالا
        self.df.dropna(subset=['ID_Customer', 'ID_Item'], inplace=True)

        # پر کردن مقادیر عددی و متنی
        self.df['Amount_Gross_Order'].fillna(self.df['Amount_Gross_Order'].median(), inplace=True)
        self.df['Quantity_item'].fillna(1, inplace=True)
        self.df['city_name_fa'].fillna('نامشخص', inplace=True)

    def extract_datetime_features(self):
        self.df['DateTime_CartFinalize'] = pd.to_datetime(self.df['DateTime_CartFinalize'], errors='coerce')
        self.df.dropna(subset=['DateTime_CartFinalize'], inplace=True)

        self.df['year'] = self.df['DateTime_CartFinalize'].dt.year
        self.df['month'] = self.df['DateTime_CartFinalize'].dt.month
        self.df['day'] = self.df['DateTime_CartFinalize'].dt.day
        self.df['hour'] = self.df['DateTime_CartFinalize'].dt.hour
        self.df['weekday'] = self.df['DateTime_CartFinalize'].dt.weekday
        self.df['is_weekend'] = self.df['weekday'].isin([4, 5]).astype(int)
        self.df['season'] = self.df['month'] % 12 // 3 + 1

    def encode_city(self):
        # استفاده از One-Hot Encoding برای شهر
        city_dummies = pd.get_dummies(self.df['city_name_fa'], prefix='city')
        self.df = pd.concat([self.df, city_dummies], axis=1)

    def create_stat_features(self):
        # ویژگی‌های رفتاری مشتری
        customer_group = self.df.groupby('ID_Customer')

        features = pd.DataFrame()
        features['order_count'] = customer_group['ID_Order'].nunique()
        features['avg_amount'] = customer_group['Amount_Gross_Order'].mean()
        features['total_amount'] = customer_group['Amount_Gross_Order'].sum()
        features['std_amount'] = customer_group['Amount_Gross_Order'].std().fillna(0)
        features['total_items'] = customer_group['Quantity_item'].sum()
        features['unique_items'] = customer_group['ID_Item'].nunique()
        features['purchase_days'] = customer_group['day'].nunique()
        features['avg_hour'] = customer_group['hour'].mean()
        features['is_weekend_ratio'] = customer_group['is_weekend'].mean()

        self.customer_features = features.reset_index()

    def transform(self):
        self.handle_missing_values()
        self.extract_datetime_features()
        self.encode_city()
        self.create_stat_features()

        return self.customer_features
