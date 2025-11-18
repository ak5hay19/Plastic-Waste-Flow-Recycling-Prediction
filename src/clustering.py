"""
Clustering and Anomaly Detection Module

This module implements:
- K-Means clustering for country grouping
- Hierarchical clustering
- DBSCAN for density-based clustering
- Anomaly detection using multiple methods
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats


class ClusterAnalyzer:
    """
    Performs clustering analysis on plastic waste data.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.cluster_labels = {}
    
    def prepare_clustering_data(self, df, features):
        """
        Prepare and scale data for clustering.
        
        Args:
            df (pd.DataFrame): Input dataframe
            features (list): List of feature columns
        
        Returns:
            tuple: (scaled_data, scaler, original_data)
        """
        # Extract features
        X = df[features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, scaler, X
    
    def kmeans_clustering(self, df, features, n_clusters=5, random_state=42):
        """
        Perform K-Means clustering.
        
        Args:
            df (pd.DataFrame): Input dataframe
            features (list): Feature columns
            n_clusters (int): Number of clusters
            random_state (int): Random seed
        
        Returns:
            pd.DataFrame: DataFrame with cluster assignments
        """
        print(f"\n{'='*60}")
        print(f"K-Means Clustering (k={n_clusters})")
        print(f"{'='*60}")
        
        # Prepare data
        X_scaled, scaler, X_original = self.prepare_clustering_data(df, features)
        
        # Fit K-Means
        print(f"Fitting K-Means with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        result_df = df.copy()
        result_df['Cluster'] = clusters
        
        # Calculate cluster statistics
        print(f"\n✓ Clustering complete")
        print(f"\nCluster sizes:")
        for i in range(n_clusters):
            size = np.sum(clusters == i)
            pct = (size / len(clusters)) * 100
            print(f"  Cluster {i}: {size} countries ({pct:.1f}%)")
        
        # Store model
        self.models['kmeans'] = kmeans
        self.scalers['kmeans'] = scaler
        self.cluster_labels['kmeans'] = clusters
        
        # Calculate cluster centers (in original scale)
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)
        
        centers_df = pd.DataFrame(centers_original, columns=features)
        centers_df['Cluster'] = range(n_clusters)
        
        print(f"\nCluster centers:")
        print(centers_df)
        
        return result_df, centers_df
    
    def elbow_method(self, df, features, max_k=10):
        """
        Determine optimal number of clusters using elbow method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            features (list): Feature columns
            max_k (int): Maximum number of clusters to try
        
        Returns:
            dict: Inertias for each k
        """
        print(f"\nDetermining optimal k using elbow method...")
        
        # Prepare data
        X_scaled, _, _ = self.prepare_clustering_data(df, features)
        
        inertias = {}
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias[k] = kmeans.inertia_
            print(f"  k={k}: inertia={kmeans.inertia_:.2f}")
        
        return inertias
    
    def hierarchical_clustering(self, df, features, n_clusters=5, linkage_method='ward'):
        """
        Perform hierarchical clustering.
        
        Args:
            df (pd.DataFrame): Input dataframe
            features (list): Feature columns
            n_clusters (int): Number of clusters
            linkage_method (str): Linkage method ('ward', 'complete', 'average')
        
        Returns:
            pd.DataFrame: DataFrame with cluster assignments
        """
        print(f"\n{'='*60}")
        print(f"Hierarchical Clustering (n={n_clusters}, method={linkage_method})")
        print(f"{'='*60}")
        
        # Prepare data
        X_scaled, scaler, X_original = self.prepare_clustering_data(df, features)
        
        # Fit hierarchical clustering
        print(f"Fitting hierarchical clustering...")
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        clusters = hierarchical.fit_predict(X_scaled)
        
        # Add cluster labels
        result_df = df.copy()
        result_df['Cluster'] = clusters
        
        print(f"\n✓ Clustering complete")
        print(f"\nCluster sizes:")
        for i in range(n_clusters):
            size = np.sum(clusters == i)
            pct = (size / len(clusters)) * 100
            print(f"  Cluster {i}: {size} countries ({pct:.1f}%)")
        
        # Store model
        self.models['hierarchical'] = hierarchical
        self.scalers['hierarchical'] = scaler
        self.cluster_labels['hierarchical'] = clusters
        
        return result_df
    
    def dbscan_clustering(self, df, features, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering (density-based).
        
        Args:
            df (pd.DataFrame): Input dataframe
            features (list): Feature columns
            eps (float): Maximum distance between samples
            min_samples (int): Minimum samples in neighborhood
        
        Returns:
            pd.DataFrame: DataFrame with cluster assignments
        """
        print(f"\n{'='*60}")
        print(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
        print(f"{'='*60}")
        
        # Prepare data
        X_scaled, scaler, X_original = self.prepare_clustering_data(df, features)
        
        # Fit DBSCAN
        print(f"Fitting DBSCAN...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        # Add cluster labels (-1 represents noise/outliers)
        result_df = df.copy()
        result_df['Cluster'] = clusters
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        print(f"\n✓ Clustering complete")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        
        print(f"\nCluster sizes:")
        for i in set(clusters):
            size = np.sum(clusters == i)
            pct = (size / len(clusters)) * 100
            if i == -1:
                print(f"  Noise: {size} countries ({pct:.1f}%)")
            else:
                print(f"  Cluster {i}: {size} countries ({pct:.1f}%)")
        
        # Store model
        self.models['dbscan'] = dbscan
        self.scalers['dbscan'] = scaler
        self.cluster_labels['dbscan'] = clusters
        
        return result_df
    
    def pca_analysis(self, df, features, n_components=2):
        """
        Perform PCA for dimensionality reduction and visualization.
        
        Args:
            df (pd.DataFrame): Input dataframe
            features (list): Feature columns
            n_components (int): Number of components
        
        Returns:
            pd.DataFrame: DataFrame with PCA components
        """
        print(f"\nPerforming PCA with {n_components} components...")
        
        # Prepare data
        X_scaled, scaler, X_original = self.prepare_clustering_data(df, features)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create result dataframe
        result_df = df.copy()
        for i in range(n_components):
            result_df[f'PC{i+1}'] = X_pca[:, i]
        
        # Print explained variance
        print(f"\n✓ PCA complete")
        print(f"Explained variance ratio:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {var*100:.2f}%")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        self.models['pca'] = pca
        
        return result_df, pca


class AnomalyDetector:
    """
    Detects anomalies in plastic waste data using multiple methods.
    """
    
    def __init__(self):
        self.models = {}
        self.anomaly_scores = {}
    
    def isolation_forest_detection(self, df, features, contamination=0.1):
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            df (pd.DataFrame): Input dataframe
            features (list): Feature columns
            contamination (float): Expected proportion of outliers
        
        Returns:
            pd.DataFrame: DataFrame with anomaly labels
        """
        print(f"\n{'='*60}")
        print(f"Isolation Forest Anomaly Detection")
        print(f"{'='*60}")
        
        # Prepare data
        X = df[features].fillna(df[features].mean())
        
        # Fit Isolation Forest
        print(f"Fitting Isolation Forest (contamination={contamination})...")
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.score_samples(X)
        
        # Add to dataframe (-1 = anomaly, 1 = normal)
        result_df = df.copy()
        result_df['Anomaly_IF'] = anomaly_labels
        result_df['Anomaly_Score_IF'] = anomaly_scores
        result_df['Is_Anomaly_IF'] = (anomaly_labels == -1)
        
        n_anomalies = np.sum(anomaly_labels == -1)
        pct_anomalies = (n_anomalies / len(anomaly_labels)) * 100
        
        print(f"\n✓ Detection complete")
        print(f"  Anomalies detected: {n_anomalies} ({pct_anomalies:.2f}%)")
        
        # Show top anomalies
        if 'Entity' in result_df.columns:
            print(f"\nTop 10 anomalies:")
            anomalies = result_df[result_df['Is_Anomaly_IF']].nsmallest(10, 'Anomaly_Score_IF')
            print(anomalies[['Entity'] + features + ['Anomaly_Score_IF']])
        
        self.models['isolation_forest'] = iso_forest
        self.anomaly_scores['isolation_forest'] = anomaly_scores
        
        return result_df
    
    def statistical_outlier_detection(self, df, features, threshold=3):
        """
        Detect outliers using statistical methods (Z-score).
        
        Args:
            df (pd.DataFrame): Input dataframe
            features (list): Feature columns
            threshold (float): Z-score threshold
        
        Returns:
            pd.DataFrame: DataFrame with outlier labels
        """
        print(f"\n{'='*60}")
        print(f"Statistical Outlier Detection (Z-score > {threshold})")
        print(f"{'='*60}")
        
        result_df = df.copy()
        
        # Calculate Z-scores for each feature
        for feature in features:
            values = df[feature].fillna(df[feature].mean())
            z_scores = np.abs(stats.zscore(values))
            result_df[f'Z_score_{feature}'] = z_scores
        
        # Identify outliers (any feature with Z-score > threshold)
        z_score_cols = [f'Z_score_{f}' for f in features]
        result_df['Max_Z_score'] = result_df[z_score_cols].max(axis=1)
        result_df['Is_Outlier_Statistical'] = result_df['Max_Z_score'] > threshold
        
        n_outliers = result_df['Is_Outlier_Statistical'].sum()
        pct_outliers = (n_outliers / len(result_df)) * 100
        
        print(f"\n✓ Detection complete")
        print(f"  Outliers detected: {n_outliers} ({pct_outliers:.2f}%)")
        
        if 'Entity' in result_df.columns and n_outliers > 0:
            print(f"\nOutliers:")
            outliers = result_df[result_df['Is_Outlier_Statistical']]
            print(outliers[['Entity'] + features + ['Max_Z_score']])
        
        return result_df
    
    def trade_flow_anomalies(self, trade_df, threshold_quantile=0.95):
        """
        Detect anomalous trade flows.
        
        Args:
            trade_df (pd.DataFrame): Trade data
            threshold_quantile (float): Quantile threshold for large trades
        
        Returns:
            pd.DataFrame: Anomalous trade flows
        """
        print(f"\n{'='*60}")
        print(f"Trade Flow Anomaly Detection")
        print(f"{'='*60}")
        
        # Detect unusually large flows
        threshold = trade_df['Quantity_Tonnes'].quantile(threshold_quantile)
        
        large_flows = trade_df[trade_df['Quantity_Tonnes'] > threshold].copy()
        
        print(f"\n✓ Detection complete")
        print(f"  Threshold (P{threshold_quantile*100}): {threshold:,.2f} tonnes")
        print(f"  Anomalous flows: {len(large_flows)}")
        
        if len(large_flows) > 0:
            print(f"\nTop 10 largest flows:")
            top_flows = large_flows.nlargest(10, 'Quantity_Tonnes')
            if 'Reporter' in top_flows.columns:
                print(top_flows[['Reporter', 'Partner', 'Year', 'Quantity_Tonnes', 'Value_USD']])
        
        # Detect unusual value patterns
        trade_df_clean = trade_df[trade_df['Value_Per_Kg_USD'].notna()].copy()
        
        # Z-score for value per kg
        z_scores = np.abs(stats.zscore(trade_df_clean['Value_Per_Kg_USD']))
        unusual_value = trade_df_clean[z_scores > 3].copy()
        
        print(f"\n  Unusual pricing detected: {len(unusual_value)} flows")
        
        return {
            'large_flows': large_flows,
            'unusual_pricing': unusual_value
        }
    
    def time_series_anomalies(self, timeseries_df, column='Production_Tonnes', window=3):
        """
        Detect anomalies in time series data.
        
        Args:
            timeseries_df (pd.DataFrame): Time series data
            column (str): Column to analyze
            window (int): Rolling window size
        
        Returns:
            pd.DataFrame: Time series with anomaly flags
        """
        print(f"\n{'='*60}")
        print(f"Time Series Anomaly Detection")
        print(f"{'='*60}")
        
        result_df = timeseries_df.copy()
        
        # Calculate rolling mean and std
        rolling_mean = result_df[column].rolling(window=window, center=True).mean()
        rolling_std = result_df[column].rolling(window=window, center=True).std()
        
        # Calculate Z-score relative to rolling statistics
        z_scores = np.abs((result_df[column] - rolling_mean) / rolling_std)
        
        result_df['Rolling_Mean'] = rolling_mean
        result_df['Rolling_Std'] = rolling_std
        result_df['Z_Score'] = z_scores
        result_df['Is_Anomaly'] = z_scores > 3
        
        n_anomalies = result_df['Is_Anomaly'].sum()
        
        print(f"\n✓ Detection complete")
        print(f"  Anomalies detected: {n_anomalies}")
        
        if n_anomalies > 0:
            print(f"\nAnomalous periods:")
            anomalies = result_df[result_df['Is_Anomaly']]
            print(anomalies[['Year', column, 'Rolling_Mean', 'Z_Score']])
        
        return result_df


if __name__ == "__main__":
    # Test clustering and anomaly detection
    from data_loader import load_data
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    data = load_data()
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_all(data)
    
    # Initialize analyzers
    cluster_analyzer = ClusterAnalyzer()
    anomaly_detector = AnomalyDetector()
    
    # Clustering on waste data
    print("\n" + "="*70)
    print("CLUSTERING ANALYSIS")
    print("="*70)
    
    features = ['Waste_Per_Capita_kg']
    
    # K-Means
    clustered_df, centers = cluster_analyzer.kmeans_clustering(
        processed['waste_countries'],
        features,
        n_clusters=5
    )
    
    # Anomaly detection on waste data
    print("\n" + "="*70)
    print("ANOMALY DETECTION")
    print("="*70)
    
    # Isolation Forest
    anomaly_df = anomaly_detector.isolation_forest_detection(
        processed['waste_countries'],
        features,
        contamination=0.1
    )
    
    # Statistical outliers
    outlier_df = anomaly_detector.statistical_outlier_detection(
        processed['waste_countries'],
        features,
        threshold=3
    )

