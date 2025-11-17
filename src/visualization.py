"""
Visualization Module

This module creates various visualizations for plastic waste analysis including:
- Time series plots
- Geographic maps
- Network graphs
- Statistical plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class Visualizer:
    """
    Creates visualizations for plastic waste data analysis.
    """
    
    def __init__(self, output_dir='outputs/figures'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_production_trend(self, production_df, save=True):
        """
        Plot global plastic production trend over time.
        
        Args:
            production_df (pd.DataFrame): Production data
            save (bool): Whether to save the figure
        
        Returns:
            plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=production_df['Year'],
            y=production_df['Production_Million_Tonnes'],
            mode='lines+markers',
            name='Production',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Global Plastic Production (1950-2019)',
            xaxis_title='Year',
            yaxis_title='Production (Million Tonnes)',
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        if save:
            fig.write_html(f'{self.output_dir}/production_trend.html')
            print(f"✓ Saved: production_trend.html")
        
        return fig
    
    def plot_production_growth(self, production_df, save=True):
        """
        Plot year-over-year growth rate.
        
        Args:
            production_df (pd.DataFrame): Production data
            save (bool): Whether to save the figure
        
        Returns:
            plotly figure
        """
        fig = go.Figure()
        
        # Filter out NaN values
        df_clean = production_df[production_df['YoY_Change'].notna()]
        
        fig.add_trace(go.Bar(
            x=df_clean['Year'],
            y=df_clean['YoY_Change'],
            marker_color=np.where(df_clean['YoY_Change'] > 0, '#06A77D', '#D90368'),
            name='YoY Growth'
        ))
        
        fig.update_layout(
            title='Year-over-Year Production Growth Rate',
            xaxis_title='Year',
            yaxis_title='Growth Rate (%)',
            template='plotly_white',
            height=500
        )
        
        if save:
            fig.write_html(f'{self.output_dir}/production_growth.html')
            print(f"✓ Saved: production_growth.html")
        
        return fig
    
    def plot_waste_choropleth(self, waste_df, save=True):
        """
        Create choropleth map of mismanaged waste per capita.
        
        Args:
            waste_df (pd.DataFrame): Waste data
            save (bool): Whether to save the figure
        
        Returns:
            plotly figure
        """
        fig = px.choropleth(
            waste_df,
            locations='Code',
            color='Waste_Per_Capita_kg',
            hover_name='Entity',
            hover_data=['Waste_Per_Capita_kg'],
            color_continuous_scale='Reds',
            title='Mismanaged Plastic Waste Per Capita (2019)',
            labels={'Waste_Per_Capita_kg': 'Waste (kg/year)'}
        )
        
        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True),
            height=600
        )
        
        if save:
            fig.write_html(f'{self.output_dir}/waste_choropleth.html')
            print(f"✓ Saved: waste_choropleth.html")
        
        return fig
    
    def plot_top_countries_waste(self, waste_df, n=20, save=True):
        """
        Plot top countries by mismanaged waste.
        
        Args:
            waste_df (pd.DataFrame): Waste data
            n (int): Number of countries to show
            save (bool): Whether to save the figure
        
        Returns:
            plotly figure
        """
        top_countries = waste_df.nlargest(n, 'Waste_Per_Capita_kg')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_countries['Waste_Per_Capita_kg'],
            y=top_countries['Entity'],
            orientation='h',
            marker=dict(
                color=top_countries['Waste_Per_Capita_kg'],
                colorscale='Reds',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title=f'Top {n} Countries by Mismanaged Plastic Waste Per Capita',
            xaxis_title='Waste Per Capita (kg/year)',
            yaxis_title='Country',
            template='plotly_white',
            height=max(500, n * 25),
            yaxis=dict(autorange='reversed')
        )
        
        if save:
            fig.write_html(f'{self.output_dir}/top_countries_waste.html')
            print(f"✓ Saved: top_countries_waste.html")
        
        return fig
    
    def plot_trade_network(self, network_df, top_n=30, save=True):
        """
        Visualize trade network using NetworkX and Plotly.
        
        Args:
            network_df (pd.DataFrame): Network edge list
            top_n (int): Number of top flows to visualize
            save (bool): Whether to save the figure
        
        Returns:
            plotly figure
        """
        # Get top flows
        top_flows = network_df.nlargest(top_n, 'Total_Quantity_Tonnes')
        
        # Create network graph
        G = nx.DiGraph()
        
        for _, row in top_flows.iterrows():
            G.add_edge(
                row['Source'],
                row['Target'],
                weight=row['Total_Quantity_Tonnes']
            )
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            # Size based on degree
            node_size.append(G.degree(node) * 10 + 20)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=node_size,
                color='#2E86AB',
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title=f'Plastic Waste Trade Network (Top {top_n} Flows)',
            showlegend=False,
            hovermode='closest',
            template='plotly_white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700
        )
        
        if save:
            fig.write_html(f'{self.output_dir}/trade_network.html')
            print(f"✓ Saved: trade_network.html")
        
        return fig
    
    def plot_trade_flows_time(self, trade_yearly_df, save=True):
        """
        Plot trade flows over time.
        
        Args:
            trade_yearly_df (pd.DataFrame): Yearly aggregated trade data
            save (bool): Whether to save the figure
        
        Returns:
            plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Trade Volume Over Time', 'Number of Trading Partners'),
            vertical_spacing=0.15
        )
        
        # Trade volume
        fig.add_trace(
            go.Scatter(
                x=trade_yearly_df['Year'],
                y=trade_yearly_df['Total_Quantity_Tonnes'],
                mode='lines+markers',
                name='Trade Volume',
                line=dict(color='#2E86AB', width=3)
            ),
            row=1, col=1
        )
        
        # Number of traders
        fig.add_trace(
            go.Scatter(
                x=trade_yearly_df['Year'],
                y=trade_yearly_df['Num_Exporters'],
                mode='lines+markers',
                name='Exporters',
                line=dict(color='#06A77D', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=trade_yearly_df['Year'],
                y=trade_yearly_df['Num_Importers'],
                mode='lines+markers',
                name='Importers',
                line=dict(color='#D90368', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Volume (Tonnes)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Countries", row=2, col=1)
        
        fig.update_layout(
            height=800,
            template='plotly_white',
            hovermode='x unified'
        )
        
        if save:
            fig.write_html(f'{self.output_dir}/trade_flows_time.html')
            print(f"✓ Saved: trade_flows_time.html")
        
        return fig
    
    def plot_clustering_results(self, clustered_df, features, cluster_col='Cluster', save=True):
        """
        Visualize clustering results.
        
        Args:
            clustered_df (pd.DataFrame): Data with cluster labels
            features (list): Feature columns used for clustering
            cluster_col (str): Name of cluster column
            save (bool): Whether to save the figure
        
        Returns:
            plotly figure
        """
        if len(features) == 1:
            # 1D scatter plot
            fig = px.strip(
                clustered_df,
                x=features[0],
                color=cluster_col,
                hover_data=['Entity'] if 'Entity' in clustered_df.columns else None,
                title='Clustering Results'
            )
        else:
            # 2D scatter plot (use first two features or PCA components)
            fig = px.scatter(
                clustered_df,
                x=features[0],
                y=features[1] if len(features) > 1 else features[0],
                color=cluster_col,
                hover_data=['Entity'] if 'Entity' in clustered_df.columns else None,
                title='Clustering Results',
                color_continuous_scale='viridis' if clustered_df[cluster_col].dtype != 'object' else None
            )
        
        fig.update_layout(template='plotly_white', height=600)
        
        if save:
            fig.write_html(f'{self.output_dir}/clustering_results.html')
            print(f"✓ Saved: clustering_results.html")
        
        return fig
    
    def plot_forecast_comparison(self, historical_df, forecast_dict, column='Production_Tonnes', save=True):
        """
        Compare forecasts from multiple models.
        
        Args:
            historical_df (pd.DataFrame): Historical data
            forecast_dict (dict): Dictionary of forecasts from different models
            column (str): Column name for historical data
            save (bool): Whether to save the figure
        
        Returns:
            plotly figure
        """
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=historical_df.index,
            y=historical_df[column],
            mode='lines+markers',
            name='Historical',
            line=dict(color='black', width=2)
        ))
        
        # Plot forecasts
        colors = ['#2E86AB', '#06A77D', '#D90368', '#F77F00', '#8338EC']
        
        for i, (model_name, forecast_df) in enumerate(forecast_dict.items()):
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Forecast'],
                mode='lines+markers',
                name=model_name.upper(),
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Production Forecast Comparison',
            xaxis_title='Year',
            yaxis_title='Production (Tonnes)',
            template='plotly_white',
            hovermode='x unified',
            height=600
        )
        
        if save:
            fig.write_html(f'{self.output_dir}/forecast_comparison.html')
            print(f"✓ Saved: forecast_comparison.html")
        
        return fig
    
    def plot_anomalies(self, df, value_col, anomaly_col='Is_Anomaly', entity_col='Entity', save=True):
        """
        Visualize detected anomalies.
        
        Args:
            df (pd.DataFrame): Data with anomaly flags
            value_col (str): Value column
            anomaly_col (str): Anomaly flag column
            entity_col (str): Entity/country column
            save (bool): Whether to save the figure
        
        Returns:
            plotly figure
        """
        df_plot = df.copy()
        df_plot['Anomaly_Status'] = df_plot[anomaly_col].map({True: 'Anomaly', False: 'Normal'})
        
        fig = px.scatter(
            df_plot,
            x=df_plot.index if entity_col not in df_plot.columns else entity_col,
            y=value_col,
            color='Anomaly_Status',
            color_discrete_map={'Normal': '#2E86AB', 'Anomaly': '#D90368'},
            title='Anomaly Detection Results',
            hover_data=[value_col]
        )
        
        fig.update_layout(
            template='plotly_white',
            height=600
        )
        
        if save:
            fig.write_html(f'{self.output_dir}/anomalies.html')
            print(f"✓ Saved: anomalies.html")
        
        return fig
    
    def create_summary_dashboard(self, data_dict, save=True):
        """
        Create a comprehensive summary dashboard.
        
        Args:
            data_dict (dict): Dictionary containing all processed data
            save (bool): Whether to save the figure
        
        Returns:
            plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Global Production Trend',
                'Top 10 Countries by Mismanaged Waste',
                'Waste Distribution by Category',
                'Trade Volume by Year'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Production trend
        if 'production' in data_dict:
            prod = data_dict['production']
            fig.add_trace(
                go.Scatter(
                    x=prod['Year'],
                    y=prod['Production_Million_Tonnes'],
                    mode='lines',
                    name='Production',
                    line=dict(color='#2E86AB', width=2)
                ),
                row=1, col=1
            )
        
        # Top countries
        if 'waste_countries' in data_dict:
            waste = data_dict['waste_countries']
            top10 = waste.nlargest(10, 'Waste_Per_Capita_kg')
            fig.add_trace(
                go.Bar(
                    x=top10['Waste_Per_Capita_kg'],
                    y=top10['Entity'],
                    orientation='h',
                    marker=dict(color='#D90368'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Waste categories
        if 'waste_countries' in data_dict:
            waste = data_dict['waste_countries']
            category_counts = waste['Waste_Category'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    marker=dict(color='#06A77D'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Trade volume
        if 'trade_yearly' in data_dict:
            trade = data_dict['trade_yearly']
            fig.add_trace(
                go.Scatter(
                    x=trade['Year'],
                    y=trade['Total_Quantity_Tonnes'],
                    mode='lines+markers',
                    name='Trade Volume',
                    line=dict(color='#F77F00', width=2)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text='Plastic Waste Analytics Dashboard',
            showlegend=False,
            template='plotly_white',
            height=800
        )
        
        if save:
            fig.write_html(f'{self.output_dir}/summary_dashboard.html')
            print(f"✓ Saved: summary_dashboard.html")
        
        return fig


if __name__ == "__main__":
    # Test visualizations
    from data_loader import load_data
    from preprocessing import DataPreprocessor
    
    # Load and preprocess data
    print("Loading data...")
    data = load_data(trade_sample=0.1)
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_all(data)
    
    # Initialize visualizer
    viz = Visualizer()
    
    print("\nCreating visualizations...")
    
    # Production plots
    viz.plot_production_trend(processed['production'])
    viz.plot_production_growth(processed['production'])
    
    # Waste plots
    viz.plot_waste_choropleth(processed['waste_countries'])
    viz.plot_top_countries_waste(processed['waste_countries'], n=20)
    
    # Trade plots
    viz.plot_trade_network(processed['trade_network'], top_n=30)
    viz.plot_trade_flows_time(processed['trade_yearly'])
    
    # Summary dashboard
    viz.create_summary_dashboard(processed)
    
    print("\n✓ All visualizations created successfully!")

