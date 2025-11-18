"""
Flow Analysis Module

This module implements graph-based network analysis for plastic waste flows
using NetworkX and other graph analytics tools.
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class FlowAnalyzer:
    """
    Analyzes plastic waste flows as a network/graph structure.
    """
    
    def __init__(self):
        self.graph = None
        self.metrics = {}
    
    def create_network_graph(self, network_df, flow_type='Export', min_weight=None):
        """
        Create a directed graph from trade network data.
        
        Args:
            network_df (pd.DataFrame): Network edge list
            flow_type (str): 'Export' or 'Import'
            min_weight (float): Minimum edge weight to include
        
        Returns:
            nx.DiGraph: Directed graph object
        """
        print(f"Creating network graph for {flow_type} flows...")
        
        # Filter by flow type
        df = network_df[network_df['Flow_Type'] == flow_type].copy()
        
        # Apply minimum weight filter if specified
        if min_weight:
            df = df[df['Total_Quantity_Tonnes'] >= min_weight]
            print(f"  Applied minimum weight filter: {min_weight} tonnes")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        all_countries = set(df['Source'].unique()) | set(df['Target'].unique())
        for country in all_countries:
            G.add_node(country)
        
        # Add edges with weights
        for _, row in df.iterrows():
            G.add_edge(
                row['Source'],
                row['Target'],
                weight=row['Total_Quantity_Tonnes'],
                value=row['Total_Value_USD'],
                transactions=row['Num_Transactions'],
                years=f"{row['First_Year']}-{row['Last_Year']}"
            )
        
        print(f"  ✓ Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        self.graph = G
        return G
    
    def calculate_centrality_metrics(self, G=None):
        """
        Calculate various centrality metrics for the network.
        
        Args:
            G (nx.DiGraph): Graph object (uses self.graph if None)
        
        Returns:
            dict: Dictionary of centrality metrics
        """
        if G is None:
            G = self.graph
        
        if G is None:
            print("Error: No graph available. Create graph first.")
            return None
        
        print("\nCalculating centrality metrics...")
        
        metrics = {}
        
        # Degree centrality
        print("  Computing degree centrality...")
        metrics['degree_centrality'] = nx.degree_centrality(G)
        metrics['in_degree_centrality'] = nx.in_degree_centrality(G)
        metrics['out_degree_centrality'] = nx.out_degree_centrality(G)
        
        # Betweenness centrality
        print("  Computing betweenness centrality...")
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G, weight='weight')
        
        # Closeness centrality
        print("  Computing closeness centrality...")
        try:
            metrics['closeness_centrality'] = nx.closeness_centrality(G)
        except:
            print("    Warning: Could not compute closeness centrality (graph may not be connected)")
            metrics['closeness_centrality'] = {}
        
        # PageRank (weighted)
        print("  Computing PageRank...")
        metrics['pagerank'] = nx.pagerank(G, weight='weight')
        
        # Eigenvector centrality
        print("  Computing eigenvector centrality...")
        try:
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        except:
            print("    Warning: Could not compute eigenvector centrality")
            metrics['eigenvector_centrality'] = {}
        
        self.metrics = metrics
        print("  ✓ Centrality metrics calculated")
        
        return metrics
    
    def get_top_nodes(self, metric='degree_centrality', n=10):
        """
        Get top nodes by a specific centrality metric.
        
        Args:
            metric (str): Name of centrality metric
            n (int): Number of top nodes to return
        
        Returns:
            list: List of (node, value) tuples
        """
        if not self.metrics or metric not in self.metrics:
            print(f"Error: Metric '{metric}' not available. Calculate metrics first.")
            return []
        
        metric_dict = self.metrics[metric]
        top_nodes = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return top_nodes
    
    def identify_communities(self, G=None):
        """
        Identify communities/clusters in the network.
        
        Args:
            G (nx.DiGraph): Graph object (uses self.graph if None)
        
        Returns:
            dict: Community assignments
        """
        if G is None:
            G = self.graph
        
        if G is None:
            print("Error: No graph available. Create graph first.")
            return None
        
        print("\nIdentifying communities...")
        
        # Convert to undirected for community detection
        G_undirected = G.to_undirected()
        
        # Use Louvain method for community detection
        from networkx.algorithms import community
        
        communities = community.greedy_modularity_communities(G_undirected, weight='weight')
        
        # Create community assignment dictionary
        community_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_dict[node] = i
        
        print(f"  ✓ Identified {len(communities)} communities")
        
        # Show community sizes
        for i, comm in enumerate(communities):
            print(f"    Community {i}: {len(comm)} countries")
        
        return community_dict
    
    def calculate_network_stats(self, G=None):
        """
        Calculate overall network statistics.
        
        Args:
            G (nx.DiGraph): Graph object (uses self.graph if None)
        
        Returns:
            dict: Network statistics
        """
        if G is None:
            G = self.graph
        
        if G is None:
            print("Error: No graph available. Create graph first.")
            return None
        
        print("\nCalculating network statistics...")
        
        # Check if graph is empty
        if G.number_of_nodes() == 0:
            print("  Warning: Graph is empty (no nodes)")
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'density': 0,
                'is_connected': False,
                'num_connected_components': 0,
                'avg_degree': 0,
                'max_degree': 0,
                'min_degree': 0,
                'avg_shortest_path_length': None,
                'avg_clustering_coefficient': 0,
                'total_flow_tonnes': 0
            }
        
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
            'num_connected_components': nx.number_weakly_connected_components(G) if G.number_of_nodes() > 0 else 0
        }
        
        # Average degree
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = max(degrees)
        stats['min_degree'] = min(degrees)
        
        # Average path length (for largest component only)
        try:
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            stats['avg_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
        except:
            stats['avg_shortest_path_length'] = None
        
        # Clustering coefficient
        G_undirected = G.to_undirected()
        stats['avg_clustering_coefficient'] = nx.average_clustering(G_undirected)
        
        # Total flow
        total_flow = sum(data['weight'] for u, v, data in G.edges(data=True))
        stats['total_flow_tonnes'] = total_flow
        
        print("  ✓ Network statistics calculated")
        for key, value in stats.items():
            print(f"    {key}: {value}")
        
        return stats
    
    def find_critical_nodes(self, G=None, top_n=10):
        """
        Identify critical nodes whose removal would most impact the network.
        
        Args:
            G (nx.DiGraph): Graph object (uses self.graph if None)
            top_n (int): Number of critical nodes to identify
        
        Returns:
            list: List of critical nodes
        """
        if G is None:
            G = self.graph
        
        if G is None:
            print("Error: No graph available. Create graph first.")
            return None
        
        print(f"\nIdentifying {top_n} critical nodes...")
        
        # Use betweenness centrality as proxy for criticality
        betweenness = nx.betweenness_centrality(G, weight='weight')
        critical = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        critical_nodes = [node for node, score in critical]
        
        print("  ✓ Critical nodes identified:")
        for i, (node, score) in enumerate(critical, 1):
            print(f"    {i}. {node}: {score:.4f}")
        
        return critical_nodes
    
    def analyze_flow_patterns(self, trade_df):
        """
        Analyze patterns in trade flows over time.
        
        Args:
            trade_df (pd.DataFrame): Trade data
        
        Returns:
            dict: Flow pattern analysis results
        """
        print("\nAnalyzing flow patterns...")
        
        patterns = {}
        
        # Flow by year
        yearly_flows = trade_df.groupby(['Year', 'Flow_Type'])['Quantity_Tonnes'].sum().unstack(fill_value=0)
        patterns['yearly_flows'] = yearly_flows
        
        # Top corridors (source-destination pairs)
        corridors = trade_df.groupby(['Reporter', 'Partner'])['Quantity_Tonnes'].sum().reset_index()
        corridors = corridors.nlargest(20, 'Quantity_Tonnes')
        patterns['top_corridors'] = corridors
        
        # Regional analysis
        # Define regions (simplified)
        asia_countries = ['China', 'India', 'Japan', 'South Korea', 'Thailand', 'Malaysia', 
                         'Indonesia', 'Vietnam', 'Philippines', 'Singapore']
        europe_countries = ['Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'Belgium', 
                           'United Kingdom', 'Poland', 'Austria', 'Sweden']
        americas_countries = ['USA', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Chile']
        
        def get_region(country):
            if country in asia_countries:
                return 'Asia'
            elif country in europe_countries:
                return 'Europe'
            elif country in americas_countries:
                return 'Americas'
            else:
                return 'Other'
        
        trade_df['Reporter_Region'] = trade_df['Reporter'].apply(get_region)
        trade_df['Partner_Region'] = trade_df['Partner'].apply(get_region)
        
        regional_flows = trade_df.groupby(['Reporter_Region', 'Partner_Region'])['Quantity_Tonnes'].sum().reset_index()
        patterns['regional_flows'] = regional_flows
        
        print("  ✓ Flow patterns analyzed")
        
        return patterns
    
    def export_to_dataframe(self, metrics_dict=None):
        """
        Export centrality metrics to a pandas DataFrame.
        
        Args:
            metrics_dict (dict): Metrics dictionary (uses self.metrics if None)
        
        Returns:
            pd.DataFrame: Metrics in tabular format
        """
        if metrics_dict is None:
            metrics_dict = self.metrics
        
        if not metrics_dict:
            print("Error: No metrics available. Calculate metrics first.")
            return None
        
        # Create dataframe
        df_list = []
        
        for metric_name, metric_values in metrics_dict.items():
            for node, value in metric_values.items():
                df_list.append({
                    'Country': node,
                    'Metric': metric_name,
                    'Value': value
                })
        
        df = pd.DataFrame(df_list)
        df_pivot = df.pivot(index='Country', columns='Metric', values='Value')
        df_pivot = df_pivot.reset_index()
        
        return df_pivot


if __name__ == "__main__":
    # Test flow analysis
    from data_loader import load_data
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    data = load_data(trade_sample=0.2)
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_all(data)
    
    # Perform flow analysis
    analyzer = FlowAnalyzer()
    
    # Create network
    G = analyzer.create_network_graph(
        processed['trade_network'], 
        flow_type='Export',
        min_weight=1000  # Only edges with > 1000 tonnes
    )
    
    # Calculate metrics
    metrics = analyzer.calculate_centrality_metrics(G)
    
    # Get top countries
    print("\nTop 10 countries by PageRank:")
    top_pr = analyzer.get_top_nodes('pagerank', 10)
    for i, (country, score) in enumerate(top_pr, 1):
        print(f"{i}. {country}: {score:.4f}")
    
    # Network stats
    stats = analyzer.calculate_network_stats(G)
    
    # Critical nodes
    critical = analyzer.find_critical_nodes(G, 10)

