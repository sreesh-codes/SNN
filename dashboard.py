import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
from snn_dataset_generator import SNNDatasetGenerator
import zipfile
import io

# Set page configuration
st.set_page_config(
    page_title="SNN Dataset Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("🧠 SNN Dataset Generator & Visualizer")
st.markdown("""
This dashboard allows you to generate, visualize, and download datasets for **Spiking Neural Network (SNN)** research, 
specifically tailored for vector embeddings.
""")

# Sidebar Configuration
st.sidebar.header("Dataset Configuration")

# Dataset Type Selection
dataset_type = st.sidebar.selectbox(
    "Select Dataset Type",
    [
        "Classification (Rate Encoding)",
        "Classification (Temporal Encoding)",
        "Clustering",
        "Temporal Patterns"
    ]
)

# Common Parameters
st.sidebar.subheader("Parameters")
n_samples = st.sidebar.number_input("Number of Samples", min_value=10, max_value=10000, value=100, step=50)
random_seed = st.sidebar.number_input("Random Seed", value=42)

# Dynamic Parameters based on Type
params = {}
if "Classification" in dataset_type:
    n_features = st.sidebar.number_input("Number of Features", min_value=2, max_value=1000, value=20)
    n_classes = st.sidebar.number_input("Number of Classes", min_value=2, max_value=20, value=5)
    max_time_steps = st.sidebar.number_input("Time Steps", min_value=10, max_value=1000, value=100)
    
    params = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'max_time_steps': max_time_steps
    }
    
    if "Rate" in dataset_type:
        params['encoding'] = 'rate'
    else:
        params['encoding'] = 'temporal'
        
elif dataset_type == "Clustering":
    n_features = st.sidebar.number_input("Number of Features", min_value=2, max_value=1000, value=15)
    n_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, max_value=20, value=4)
    cluster_std = st.sidebar.slider("Cluster Separation (Std Dev)", 0.1, 5.0, 1.5)
    max_time_steps = st.sidebar.number_input("Time Steps", min_value=10, max_value=1000, value=100)
    
    params = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_clusters': n_clusters,
        'cluster_std': cluster_std,
        'max_time_steps': max_time_steps,
        'encoding': 'rate'
    }

elif dataset_type == "Temporal Patterns":
    n_features = st.sidebar.number_input("Number of Features", min_value=2, max_value=1000, value=10)
    n_classes = st.sidebar.number_input("Number of Patterns", min_value=2, max_value=5, value=5)
    pattern_length = st.sidebar.number_input("Pattern Length (Time Steps)", min_value=10, max_value=500, value=50)
    
    params = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'pattern_length': pattern_length
    }

# Helper function for visualization (adapted for Streamlit)
def create_spike_raster_plot(dataset, n_samples=5):
    spike_trains = dataset['spike_trains']
    labels = dataset['labels']
    
    # Determine number of samples to plot
    n_plot = min(n_samples, len(spike_trains))
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2 * n_plot))
    if n_plot == 1:
        axes = [axes]
    
    for i in range(n_plot):
        ax = axes[i]
        
        # Plot spike raster
        spike_times, neuron_ids = np.where(spike_trains[i].T)
        ax.scatter(spike_times, neuron_ids, s=5, c='black', marker='|')
        ax.set_ylabel(f'Neuron ID\n(Class {labels[i]})')
        ax.set_xlim(0, spike_trains.shape[2])
        ax.set_ylim(-0.5, spike_trains.shape[1] - 0.5)
        ax.grid(True, alpha=0.3)
        
        if i == n_plot - 1:
            ax.set_xlabel('Time Step')
        else:
            ax.set_xticks([])
            
    plt.tight_layout()
    return fig

# Main Action
if st.sidebar.button("🚀 Generate Dataset", type="primary"):
    with st.spinner("Generating dataset..."):
        try:
            # Initialize Generator
            generator = SNNDatasetGenerator(random_state=random_seed)
            
            # Generate specific dataset
            if "Classification" in dataset_type:
                dataset = generator.generate_classification_dataset(**params)
            elif dataset_type == "Clustering":
                dataset = generator.generate_clustering_dataset(**params)
            elif dataset_type == "Temporal Patterns":
                dataset = generator.generate_temporal_pattern_dataset(**params)
            
            st.session_state['dataset'] = dataset
            st.session_state['dataset_type'] = dataset_type
            st.session_state['params'] = params
            st.success("Dataset generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating dataset: {str(e)}")

# Display Results if dataset exists in session state
if 'dataset' in st.session_state:
    dataset = st.session_state['dataset']
    
    # 1. Dataset Statistics
    st.header("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metadata = dataset['metadata']
    
    with col1:
        st.metric("Samples", metadata['n_samples'])
    with col2:
        st.metric("Features", metadata['n_features'])
    with col3:
        st.metric("Time Steps", metadata.get('max_time_steps', metadata.get('pattern_length')))
    with col4:
        sparsity = (1 - (dataset['spike_trains'].sum() / dataset['spike_trains'].size)) * 100
        st.metric("Sparsity", f"{sparsity:.2f}%")
        
    # 2. Visualization
    st.header("👁️ Spike Train Visualization")
    st.markdown("Below are raster plots for the first few samples. Each row represents a neuron, and dots represent spike times.")
    
    num_viz = st.slider("Number of samples to visualize", 1, 10, 3)
    fig = create_spike_raster_plot(dataset, n_samples=num_viz)
    st.pyplot(fig)
    
    # 3. Download Section
    st.header("💾 Download Dataset")
    
    # Prepare zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        # Save spike trains
        spike_buffer = io.BytesIO()
        np.save(spike_buffer, dataset['spike_trains'])
        zip_file.writestr("spike_trains.npy", spike_buffer.getvalue())
        
        # Save labels
        label_buffer = io.BytesIO()
        np.save(label_buffer, dataset['labels'])
        zip_file.writestr("labels.npy", label_buffer.getvalue())
        
        # Save features if available
        if 'features' in dataset:
            feature_buffer = io.BytesIO()
            np.save(feature_buffer, dataset['features'])
            zip_file.writestr("features.npy", feature_buffer.getvalue())
            
        # Save metadata
        zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
    
    st.download_button(
        label="Download Dataset (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"snn_dataset_{st.session_state['dataset_type'].lower().replace(' ', '_')}.zip",
        mime="application/zip"
    )
    
    # 4. Raw Data Preview
    with st.expander("See Raw Data Structure"):
        st.write("Metadata:", metadata)
        st.write(f"Spike Trains Shape: {dataset['spike_trains'].shape}")
        st.write(f"Labels Shape: {dataset['labels'].shape}")

else:
    st.info("👈 Configure parameters in the sidebar and click 'Generate Dataset' to start.")

