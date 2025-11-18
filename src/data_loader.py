"""
Data Loading Module

This module handles loading and initial processing of the three main datasets:
1. Global plastics production (time series)
2. Mismanaged plastic waste per capita (country-level)
3. UN Comtrade plastic waste trade flows
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Handles loading and initial validation of all project datasets.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize DataLoader with path to data directory.
        
        Args:
            data_dir (str): Path to directory containing CSV files
        """
        if data_dir is None:
            # Default to parent directory (where CSVs are located)
            self.data_dir = Path(__file__).parent.parent
        else:
            self.data_dir = Path(data_dir)
    
    def load_production_data(self):
        """
        Load global plastics production time series data.
        
        Returns:
            pd.DataFrame: Production data with columns [Entity, Code, Year, Production]
        """
        file_path = self.data_dir / "1- global-plastics-production.csv"
        
        try:
            df = pd.read_csv(file_path)
            # Rename for clarity
            df.columns = ['Entity', 'Code', 'Year', 'Production_Tonnes']
            
            print(f"✓ Loaded production data: {len(df)} records")
            print(f"  Year range: {df['Year'].min()} - {df['Year'].max()}")
            print(f"  Production range: {df['Production_Tonnes'].min():,.0f} - {df['Production_Tonnes'].max():,.0f} tonnes")
            
            return df
        
        except FileNotFoundError:
            print(f"✗ Error: Could not find {file_path}")
            return None
    
    def load_mismanaged_waste_data(self):
        """
        Load country-level mismanaged plastic waste per capita data.
        
        Returns:
            pd.DataFrame: Waste data with columns [Entity, Code, Year, Waste_Per_Capita_kg]
        """
        file_path = self.data_dir / "4- mismanaged-plastic-waste-per-capita.csv"
        
        try:
            df = pd.read_csv(file_path)
            # Rename for clarity
            df.columns = ['Entity', 'Code', 'Year', 'Waste_Per_Capita_kg']
            
            print(f"\n✓ Loaded mismanaged waste data: {len(df)} countries/regions")
            print(f"  Year: {df['Year'].unique()}")
            print(f"  Waste per capita range: {df['Waste_Per_Capita_kg'].min():.2f} - {df['Waste_Per_Capita_kg'].max():.2f} kg/year")
            
            return df
        
        except FileNotFoundError:
            print(f"✗ Error: Could not find {file_path}")
            return None
    
    def load_trade_data(self, sample_frac=None):
        """
        Load UN Comtrade plastic waste trade flow data.
        
        Args:
            sample_frac (float): If provided, returns a random sample of the data (0.0 to 1.0)
        
        Returns:
            pd.DataFrame: Trade flow data
        """
        file_path = self.data_dir / "comtrade_3915.csv"
        
        try:
            # Load data with proper handling - CSV has column mismatch issues
            # Use Python's csv module to properly parse, then convert to DataFrame
            import csv
            
            try:
                # Try UTF-8 first
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data_list = list(reader)
                df = pd.DataFrame(data_list)
            except UnicodeDecodeError:
                # Fall back to latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    reader = csv.DictReader(f)
                    data_list = list(reader)
                df = pd.DataFrame(data_list)
            
            # Convert numeric columns
            numeric_cols = ['refYear', 'refMonth', 'reporterCode', 'partnerCode', 'qty', 'netWgt', 'primaryValue']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if sample_frac and 0 < sample_frac < 1:
                df = df.sample(frac=sample_frac, random_state=42)
                print(f"\n✓ Loaded trade data (sample): {len(df)} records ({sample_frac*100:.0f}% of total)")
            else:
                print(f"\n✓ Loaded trade data: {len(df)} records")
            
            print(f"  Columns: {len(df.columns)}")
            
            # Try to get proper year info
            try:
                if 'period' in df.columns:
                    year_col = 'period'
                elif 'refYear' in df.columns:
                    year_col = 'refYear'
                else:
                    year_col = df.columns[0]
                
                years = df[year_col].astype(str)
                # Extract 4-digit years
                years_clean = years.str.extract(r'(\d{4})')[0]
                if years_clean.notna().any():
                    years_clean = years_clean.dropna().astype(int)
                    print(f"  Year range: {years_clean.min()} - {years_clean.max()}")
                else:
                    print(f"  Year range: {df[year_col].min()} - {df[year_col].max()}")
            except:
                print(f"  Year info: Unable to determine")
            
            print(f"  Unique reporters: {df['reporterDesc'].nunique()}")
            print(f"  Unique partners: {df['partnerDesc'].nunique()}")
            
            return df
        
        except FileNotFoundError:
            print(f"✗ Error: Could not find {file_path}")
            return None
    
    def load_all_data(self, trade_sample_frac=None):
        """
        Load all datasets at once.
        
        Args:
            trade_sample_frac (float): Optional sampling fraction for trade data
        
        Returns:
            dict: Dictionary containing all three datasets
        """
        print("=" * 70)
        print("LOADING ALL DATASETS")
        print("=" * 70)
        
        data = {
            'production': self.load_production_data(),
            'mismanaged_waste': self.load_mismanaged_waste_data(),
            'trade': self.load_trade_data(sample_frac=trade_sample_frac)
        }
        
        print("\n" + "=" * 70)
        print("ALL DATASETS LOADED SUCCESSFULLY")
        print("=" * 70)
        
        return data
    
    def get_summary_statistics(self, data_dict):
        """
        Generate summary statistics for all datasets.
        
        Args:
            data_dict (dict): Dictionary of dataframes from load_all_data()
        
        Returns:
            dict: Summary statistics
        """
        summary = {}
        
        # Production summary
        if data_dict['production'] is not None:
            prod = data_dict['production']
            summary['production'] = {
                'total_records': len(prod),
                'year_range': (int(prod['Year'].min()), int(prod['Year'].max())),
                'production_range_tonnes': (int(prod['Production_Tonnes'].min()), 
                                           int(prod['Production_Tonnes'].max())),
                'avg_annual_growth_rate': self._calculate_growth_rate(prod)
            }
        
        # Mismanaged waste summary
        if data_dict['mismanaged_waste'] is not None:
            waste = data_dict['mismanaged_waste']
            summary['mismanaged_waste'] = {
                'total_countries': len(waste),
                'year': int(waste['Year'].iloc[0]),
                'waste_range_kg': (float(waste['Waste_Per_Capita_kg'].min()), 
                                  float(waste['Waste_Per_Capita_kg'].max())),
                'global_avg_kg': float(waste[waste['Entity'] == 'World']['Waste_Per_Capita_kg'].values[0])
                                if 'World' in waste['Entity'].values else None,
                'top_5_countries': waste.nlargest(5, 'Waste_Per_Capita_kg')[['Entity', 'Waste_Per_Capita_kg']].to_dict('records')
            }
        
        # Trade summary
        if data_dict['trade'] is not None:
            trade = data_dict['trade']
            summary['trade'] = {
                'total_records': len(trade),
                'year_range': (int(trade['refYear'].min()), int(trade['refYear'].max())),
                'unique_reporters': int(trade['reporterDesc'].nunique()),
                'unique_partners': int(trade['partnerDesc'].nunique()),
                'total_quantity_kg': float(trade['qty'].sum()) if 'qty' in trade.columns else None,
                'total_value_usd': float(trade['primaryValue'].sum()) if 'primaryValue' in trade.columns else None
            }
        
        return summary
    
    @staticmethod
    def _calculate_growth_rate(prod_df):
        """Calculate average annual growth rate of plastic production."""
        first_val = prod_df.iloc[0]['Production_Tonnes']
        last_val = prod_df.iloc[-1]['Production_Tonnes']
        years = prod_df.iloc[-1]['Year'] - prod_df.iloc[0]['Year']
        
        if years > 0 and first_val > 0:
            growth_rate = ((last_val / first_val) ** (1 / years) - 1) * 100
            return round(growth_rate, 2)
        return None


# Convenience function for quick loading
def load_data(data_dir=None, trade_sample=None):
    """
    Quick function to load all datasets.
    
    Args:
        data_dir (str): Path to data directory
        trade_sample (float): Optional sampling fraction for trade data
    
    Returns:
        dict: Dictionary containing all datasets
    """
    loader = DataLoader(data_dir)
    return loader.load_all_data(trade_sample_frac=trade_sample)


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    data = loader.load_all_data(trade_sample_frac=0.1)
    
    # Print summary
    summary = loader.get_summary_statistics(data)
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for dataset_name, stats in summary.items():
        print(f"\n{dataset_name.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

