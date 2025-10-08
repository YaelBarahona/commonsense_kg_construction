import yaml
import pandas as pd
import ast
import re
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

OUTPUT_FILE = 'data/raw_data/batch_runs/emotions_chatgpt.jsonl'
INPUT_FILE = 'data/raw_data/batch_runs/emotions_chatgpt_input.jsonl'

def load_model_descriptions(yaml_path):
    """Load model definitions from YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        model_data = yaml.safe_load(f)
    return model_data['dimensional']


def build_emotion_model_dict(emotion_data, model_data):
    """
    Build a dict mapping each emotion to its model coordinates.
    Output: {emotion: {model_name: [values]}}
    """
    result = {}

    for emotion, dims in emotion_data.items():
        result[emotion] = {}
        for model_name, model_info in model_data.items():
            qdims = model_info['quality_dimensions']
            coords = [dims[d] for d in qdims if d in dims]
            result[emotion][model_name] = coords

    return result


def analyze_distribution_hist(emotion_dict, output_dir="plots"):
    """
    Analyze and plot the distribution of values for each emotion and dimension.
    Creates histograms in range [-1, 1] and saves them as PNGs.
    
    Folder structure:
        plots/
          ├── Interest/
          │     ├── Interest_arousal.png
          │     ├── Interest_valence.png
          │     └── ...
          └── Serenity/
                ├── Serenity_arousal.png
                ├── Serenity_valence.png
                └── ...
    """
    os.makedirs(output_dir, exist_ok=True)

    for emotion, dims in emotion_dict.items():
        # Create one folder per emotion
        emotion_folder = os.path.join(output_dir, emotion)
        os.makedirs(emotion_folder, exist_ok=True)

        for dim_name, values in dims.items():
            # Create histogram
            plt.figure(figsize=(6, 4))
            plt.hist(values, bins=10, range=(-1, 1), color='steelblue', edgecolor='black', alpha=0.7)
            plt.title(f"{emotion} – {dim_name.capitalize()} Distribution")
            plt.xlabel(f"{dim_name.capitalize()} Value")
            plt.ylabel("Frequency")
            plt.xlim(-1, 1)
            plt.grid(True, linestyle='--', alpha=0.5)

            # Save directly in the emotion folder
            save_path = os.path.join(emotion_folder, f"{emotion}_{dim_name}_'histrogram.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

    print(f"✅ Plots saved in '{output_dir}/<emotion>/' structure.")


def analyze_distribution_std(emotion_dict, output_dir="plots"):
    """
    Analyze and plot mean and standard deviation of each dimension per emotion.

    Creates bar plots with error bars (mean ± std) and saves them in:
        plots/<emotion>/<emotion>_mean_std.png
    """
    os.makedirs(output_dir, exist_ok=True)

    for emotion, dims in emotion_dict.items():
        emotion_folder = os.path.join(output_dir, emotion)
        os.makedirs(emotion_folder, exist_ok=True)

        dimensions = list(dims.keys())
        means = [np.mean(dims[d]) for d in dimensions]
        stds = [np.std(dims[d]) for d in dimensions]

        # Create mean ± std bar plot
        plt.figure(figsize=(8, 4))
        plt.bar(dimensions, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black', alpha=0.8)
        plt.axhline(0, color='gray', linewidth=0.8)
        plt.ylim(-1, 1)
        plt.title(f"{emotion} – Mean ± Std across Dimensions")
        plt.ylabel("Value")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Save plot
        save_path = os.path.join(emotion_folder, f"{emotion}_mean_std.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"✅ Mean–Std plots saved in '{output_dir}/<emotion>/' structure.")

def extract_response(output_data, input_data):
    raw_data_input = pd.read_json(path_or_buf=input_data, lines=True)
    raw_data_output = pd.read_json(path_or_buf=output_data, lines=True)
    data_dict_codes = {}
    data = {}
    pattern = re.compile(r'Given the concept of\s+([A-Za-z\- ]+)', re.IGNORECASE)
    for _, row in raw_data_input.iterrows():
        user_content = row['body']['messages'][1]['content']
        match = pattern.search(user_content)
        emotion = match.group(1).strip()
        id = row['custom_id']
        data_dict_codes[id] = emotion

    for _, row in raw_data_output.iterrows():
        values = ast.literal_eval(row['response']['body']['choices'][0]['message']['content'])
        dim, val = next(iter(values.items()))
        emotion = data_dict_codes[row['custom_id']]
        try:
            data[emotion][dim].append(val)
        except KeyError:
            try: 
                data[emotion][dim] = [val]
            except KeyError:
                data[emotion] = {dim : [val]}
        pass
    return data


def construct_models(inputs):
    output = {}
    for emotion, dimensions in inputs.items():
        output[emotion] = {}
        for dim, vals in dimensions.items():
            output[emotion][dim] = np.round(np.array(vals).mean(), 3)
    return output

def plot_emotions(emotion_models, model_data, model_name):
    """
    Plot emotions for a given model (2D or 3D) using proper dimension names as axis labels.
    """
    first_emotion = next(iter(emotion_models))
    coords_example = emotion_models[first_emotion][model_name]
    dim = len(coords_example)

    if dim not in (2, 3):
        raise ValueError(f"Model '{model_name}' has {dim}D — only 2D/3D supported.")

    # Get the dimension names from YAML
    dim_names = list(model_data[model_name]['quality_dimensions'].keys())

    # Collect data
    labels, coords = [], []
    for emotion, models in emotion_models.items():
        if model_name in models:
            labels.append(emotion)
            coords.append(models[model_name])

    # 2D plot
    if dim == 2:
        fig, ax = plt.subplots()
        for label, (x, y) in zip(labels, coords):
            ax.scatter(x, y)
            ax.text(x, y, label, fontsize=8, ha='right')
        ax.set_xlabel(dim_names[0])
        ax.set_ylabel(dim_names[1])
        ax.set_title(f'{model_name} Model ({dim}D)')
        ax.grid(True)

    # 3D plot
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for label, (x, y, z) in zip(labels, coords):
            ax.scatter(x, y, z)
            ax.text(x, y, z, label, fontsize=8, ha='right')
        ax.set_xlabel(dim_names[0])
        ax.set_ylabel(dim_names[1])
        ax.set_zlabel(dim_names[2])
        ax.set_title(f'{model_name} Model ({dim}D)')

    plt.show()

def create_voronoi_graph(emotion_models, model_data, model_name='circumplex', show=True):
    """
    Create a Voronoi graph for a 2D model (e.g. Circumplex: arousal–valence).
    
    Args:
        emotion_models (dict): {emotion: {model: [coords]}}
        model_data (dict): full model descriptions from YAML
        model_name (str): model to plot (default 'circumplex')
        show (bool): whether to display the plot
        
    Returns:
        vor (Voronoi): scipy.spatial.Voronoi object
        fig, ax (matplotlib Figure and Axes)
    """
    dim_names = list(model_data[model_name]['quality_dimensions'].keys())

    # Collect 2D coordinates for this model
    coords = []
    labels = []
    for emotion, models in emotion_models.items():
        if model_name in models and len(models[model_name]) == 2:
            coords.append(models[model_name])
            labels.append(emotion)
    coords = np.array(coords)

    if coords.shape[1] != 2:
        raise ValueError(f"Model '{model_name}' must be 2D for Voronoi plotting.")

    # Build Voronoi structure
    vor = Voronoi(coords)

    # Plot if requested
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='lightgray', show_points=False)
    ax.scatter(coords[:, 0], coords[:, 1], color='blue')

    for (x, y), label in zip(coords, labels):
        ax.text(x, y, label, fontsize=8, ha='right')

    ax.set_xlabel(dim_names[0])
    ax.set_ylabel(dim_names[1])
    ax.set_title(f'{model_name.capitalize()} Model Voronoi Diagram')

    # Optional display
    if show:
        plt.show()

    return vor, fig, ax


if __name__ == '__main__':
    extracted_values = extract_response(OUTPUT_FILE, INPUT_FILE)
    #analyze_distribution_hist(extracted_values)
    # analyze_distribution_std(extracted_values)
    emotion_data = construct_models(extracted_values)
    model_data = load_model_descriptions('inputs/properties_emotions.yaml')
    emotion_models = build_emotion_model_dict(emotion_data, model_data)
    print('Available models:')
    print(f'\t{model_data.keys()}')
    plot_emotions(emotion_models, model_data, 'PAD')
    # vor, fig, ax = create_voronoi_graph(emotion_models, model_data, show=True)
    #print(extracted_values)
    #print(extracted_values)

