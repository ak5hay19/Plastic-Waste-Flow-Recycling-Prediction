"""
Data Preprocessing Module

This module handles data cleaning, transformation, and preparation
for analysis and modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Handles preprocessing and cleaning of plastic waste datasets.
    """
    
    def __init__(self):
        self.processed_data = {}
    
    def preprocess_production_data(self, df):
        """
        Clean and prepare production time series data.
        
        Args:
            df (pd.DataFrame): Raw production data
        
        Returns:
            pd.DataFrame: Cleaned production data
        """
        print("Preprocessing production data...")
        
        # Make a copy
        df_clean = df.copy()
        
        # Ensure correct data types
        df_clean['Year'] = df_clean['Year'].astype(int)
        df_clean['Production_Tonnes'] = df_clean['Production_Tonnes'].astype(float)
        
        # Remove any duplicates
        df_clean = df_clean.drop_duplicates(subset=['Year'])
        
        # Sort by year
        df_clean = df_clean.sort_values('Year').reset_index(drop=True)
        
        # Add derived features
        df_clean['Production_Million_Tonnes'] = df_clean['Production_Tonnes'] / 1_000_000
        df_clean['YoY_Change'] = df_clean['Production_Tonnes'].pct_change() * 100
        df_clean['Decade'] = (df_clean['Year'] // 10) * 10
        
        # Add cumulative production
        df_clean['Cumulative_Production'] = df_clean['Production_Tonnes'].cumsum()
        
        print(f"  ✓ Cleaned {len(df_clean)} records")
        print(f"  ✓ Added 4 derived features")
        
        return df_clean
    
    def preprocess_mismanaged_waste_data(self, df):
        """
        Clean and prepare mismanaged waste data.
        
        Args:
            df (pd.DataFrame): Raw waste data
        
        Returns:
            pd.DataFrame: Cleaned waste data
        """
        print("\nPreprocessing mismanaged waste data...")
        
        # Make a copy
        df_clean = df.copy()
        
        # Handle missing country codes
        df_clean['Code'] = df_clean['Code'].fillna('UNKNOWN')
        
        # Remove aggregated regions (keep only countries)
        regions_to_remove = ['World', 'Africa', 'Asia', 'Europe', 'North America', 'Oceania', 
                            'South America', 'OWID_WRL']
        df_countries = df_clean[~df_clean['Entity'].isin(regions_to_remove)].copy()
        
        # Keep world data separately
        df_world = df_clean[df_clean['Entity'] == 'World'].copy()
        
        # Categorize countries by waste levels
        df_countries['Waste_Category'] = pd.cut(
            df_countries['Waste_Per_Capita_kg'],
            bins=[0, 1, 5, 10, 20, 100],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Add ranking
        df_countries['Global_Rank'] = df_countries['Waste_Per_Capita_kg'].rank(ascending=False)
        
        # Normalize waste values (0-1 scale)
        max_waste = df_countries['Waste_Per_Capita_kg'].max()
        df_countries['Waste_Normalized'] = df_countries['Waste_Per_Capita_kg'] / max_waste
        
        print(f"  ✓ Cleaned {len(df_countries)} countries")
        print(f"  ✓ Removed {len(df_clean) - len(df_countries)} regional aggregates")
        print(f"  ✓ Added categorization and ranking")
        
        return df_countries, df_world
    
    def preprocess_trade_data(self, df):
        """
        Clean and prepare trade flow data.
        
        Args:
            df (pd.DataFrame): Raw trade data
        
        Returns:
            pd.DataFrame: Cleaned trade data
        """
        print("\nPreprocessing trade flow data...")
        
        # Make a copy
        df_clean = df.copy()
        
        # Select relevant columns
        # Select key columns that exist
        key_columns = []
        for col in ['refYear', 'refMonth', 'period', 'reporterCode', 'reporterISO', 'reporterDesc',
                    'flowCode', 'flowDesc', 'partnerCode', 'partnerISO', 'partnerDesc',
                    'qty', 'netWgt', 'primaryValue']:
            if col in df_clean.columns:
                key_columns.append(col)
        
        df_clean = df_clean[key_columns].copy()
        
        # Extract proper year from refYear column (now properly loaded with csv.DictReader)
        if 'refYear' in df_clean.columns:
            df_clean['Year'] = pd.to_numeric(df_clean['refYear'], errors='coerce')
        elif 'period' in df_clean.columns:
            # Period might be in format like "2012" or "201201"
            df_clean['Year'] = pd.to_numeric(df_clean['period'].astype(str).str[:4], errors='coerce')
        else:
            print("  ⚠️  Warning: Could not find year column!")
        
        # Rename for clarity
        df_clean = df_clean.rename(columns={
            'reporterDesc': 'Reporter',
            'reporterISO': 'Reporter_ISO',
            'reporterCode': 'Reporter_Code',
            'partnerDesc': 'Partner',
            'partnerISO': 'Partner_ISO',
            'partnerCode': 'Partner_Code',
            'flowDesc': 'Flow_Type',
            'qty': 'Quantity_kg',
            'netWgt': 'Net_Weight_kg',
            'primaryValue': 'Value_USD'
        })
        
        # Handle missing values in quantities
        df_clean['Quantity_kg'] = df_clean['Quantity_kg'].fillna(df_clean['Net_Weight_kg'])
        df_clean['Net_Weight_kg'] = df_clean['Net_Weight_kg'].fillna(df_clean['Quantity_kg'])
        
        # Remove records with no quantity data
        df_clean = df_clean[df_clean['Quantity_kg'] > 0].copy()
        
        # Remove world aggregates from partners
        df_clean = df_clean[df_clean['Partner'] != 'World'].copy()
        
        # Convert to tonnes for easier reading
        df_clean['Quantity_Tonnes'] = df_clean['Quantity_kg'] / 1000
        
        # Add value per kg
        df_clean['Value_Per_Kg_USD'] = df_clean['Value_USD'] / df_clean['Quantity_kg']
        df_clean['Value_Per_Kg_USD'] = df_clean['Value_Per_Kg_USD'].replace([np.inf, -np.inf], np.nan)
        
        # Remove outliers in value per kg (likely data errors)
        q1 = df_clean['Value_Per_Kg_USD'].quantile(0.01)
        q99 = df_clean['Value_Per_Kg_USD'].quantile(0.99)
        df_clean = df_clean[
            (df_clean['Value_Per_Kg_USD'] >= q1) & 
            (df_clean['Value_Per_Kg_USD'] <= q99)
        ].copy()
        
        # Add flow direction indicator
        df_clean['Is_Export'] = (df_clean['Flow_Type'] == 'Export').astype(int)
        
        print(f"  ✓ Cleaned {len(df_clean)} trade records")
        print(f"  ✓ Selected {len(key_columns)} key columns")
        print(f"  ✓ Removed outliers and missing values")
        
        return df_clean
    
    def create_trade_network_data(self, trade_df):
        """
        Create network-ready data structure from trade flows.
        
        Args:
            trade_df (pd.DataFrame): Cleaned trade data
        
        Returns:
            pd.DataFrame: Edge list for network analysis
        """
        print("\nCreating network data structure...")
        
        # Aggregate flows between country pairs
        network_df = trade_df.groupby([
            'Reporter', 'Reporter_ISO', 'Partner', 'Partner_ISO', 'Flow_Type'
        ]).agg({
            'Quantity_Tonnes': 'sum',
            'Value_USD': 'sum',
            'Year': ['min', 'max', 'count']
        }).reset_index()
        
        # Flatten column names
        network_df.columns = [
            'Source', 'Source_ISO', 'Target', 'Target_ISO', 'Flow_Type',
            'Total_Quantity_Tonnes', 'Total_Value_USD', 'First_Year', 'Last_Year', 'Num_Transactions'
        ]
        
        print(f"  ✓ Created {len(network_df)} edges")
        print(f"  ✓ Unique nodes (countries): {len(set(network_df['Source'].unique()) | set(network_df['Target'].unique()))}")
        
        return network_df
    
    def aggregate_trade_by_year(self, trade_df):
        """
        Aggregate trade data by year for time series analysis.
        
        Args:
            trade_df (pd.DataFrame): Cleaned trade data
        
        Returns:
            pd.DataFrame: Yearly aggregated data
        """
        print("\nAggregating trade data by year...")
        
        yearly_data = trade_df.groupby('Year').agg({
            'Quantity_Tonnes': 'sum',
            'Value_USD': 'sum',
            'Reporter': 'nunique',
            'Partner': 'nunique'
        }).reset_index()
        
        yearly_data.columns = [
            'Year', 'Total_Quantity_Tonnes', 'Total_Value_USD', 
            'Num_Exporters', 'Num_Importers'
        ]
        
        # Add average value per tonne
        yearly_data['Avg_Value_Per_Tonne'] = (
            yearly_data['Total_Value_USD'] / yearly_data['Total_Quantity_Tonnes']
        )
        
        print(f"  ✓ Aggregated into {len(yearly_data)} years")
        
        return yearly_data
    
    def get_top_traders(self, trade_df, n=10):
        """
        Get top exporters and importers.
        
        Args:
            trade_df (pd.DataFrame): Cleaned trade data
            n (int): Number of top countries to return
        
        Returns:
            dict: Top exporters and importers
        """
        # Top exporters
        top_exporters = trade_df[trade_df['Flow_Type'] == 'Export'].groupby('Reporter').agg({
            'Quantity_Tonnes': 'sum',
            'Value_USD': 'sum'
        }).reset_index()
        top_exporters = top_exporters.nlargest(n, 'Quantity_Tonnes')
        
        # Top importers
        top_importers = trade_df[trade_df['Flow_Type'] == 'Import'].groupby('Reporter').agg({
            'Quantity_Tonnes': 'sum',
            'Value_USD': 'sum'
        }).reset_index()
        top_importers = top_importers.nlargest(n, 'Quantity_Tonnes')
        
        return {
            'exporters': top_exporters,
            'importers': top_importers
        }
    
    def preprocess_all(self, data_dict):
        """
        Preprocess all datasets.
        
        Args:
            data_dict (dict): Dictionary containing raw data
        
        Returns:
            dict: Dictionary containing all preprocessed data
        """
        print("=" * 70)
        print("PREPROCESSING ALL DATASETS")
        print("=" * 70)
        
        processed = {}
        
        # Production data
        if data_dict.get('production') is not None:
            processed['production'] = self.preprocess_production_data(data_dict['production'])
        
        # Mismanaged waste data
        if data_dict.get('mismanaged_waste') is not None:
            countries, world = self.preprocess_mismanaged_waste_data(data_dict['mismanaged_waste'])
            processed['waste_countries'] = countries
            processed['waste_world'] = world
        
        # Trade data
        if data_dict.get('trade') is not None:
            trade_clean = self.preprocess_trade_data(data_dict['trade'])
            processed['trade'] = trade_clean
            processed['trade_network'] = self.create_trade_network_data(trade_clean)
            processed['trade_yearly'] = self.aggregate_trade_by_year(trade_clean)
            processed['top_traders'] = self.get_top_traders(trade_clean)
        
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE")
        print("=" * 70)
        
        self.processed_data = processed
        return processed
    
    def save_processed_data(self, processed_dict, output_dir='outputs'):
        """
        Save preprocessed data to CSV files.
        
        Args:
            processed_dict (dict): Dictionary of processed dataframes
            output_dir (str): Output directory path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving preprocessed data to {output_dir}/...")
        
        for name, df in processed_dict.items():
            if isinstance(df, pd.DataFrame):
                output_path = f"{output_dir}/processed_{name}.csv"
                df.to_csv(output_path, index=False)
                print(f"  ✓ Saved {name}: {len(df)} records")
        
        print("All preprocessed data saved successfully!")


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_data
    
    # Load data
    data = load_data(trade_sample=0.1)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_all(data)
    
    # Save
    preprocessor.save_processed_data(processed)

