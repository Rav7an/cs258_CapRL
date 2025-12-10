import json
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_grpo_metrics(json_path, output_image_path):
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    history = data.get('log_history', [])
    if not history:
        print("No log history found.")
        return

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(history)
    
    # Filter out entries that don't have 'step' (if any)
    df = df.dropna(subset=['step'])
    
    # Metrics to plot
    metrics = [
        ('reward', 'Mean Reward'),
        ('loss', 'Loss'),
        ('entropy', 'Entropy'),
        ('kl', 'KL Divergence')
    ]
    
    # Check which metrics are actually available
    available_metrics = [m for m in metrics if m[0] in df.columns]
    
    if not available_metrics:
        print("No relevant metrics found to plot.")
        return

    num_plots = len(available_metrics)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
    
    if num_plots == 1:
        axes = [axes]

    for ax, (metric_key, metric_label) in zip(axes, available_metrics):
        # Filter out NaNs for this specific metric
        metric_data = df.dropna(subset=[metric_key])
        
        ax.plot(metric_data['step'], metric_data[metric_key], label=metric_label)
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} over Steps')
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Step')
    plt.tight_layout()
    
    print(f"Saving plots to {output_image_path}")
    plt.savefig(output_image_path)
    plt.close()

if __name__ == "__main__":
    # Path to the trainer_state.json
    # Using the checkpoint-9632 as identified
    json_file = "output/grpo_qwen2vl/checkpoint-9632/trainer_state.json"
    output_image = "output/grpo_training_plots.png"
    
    plot_grpo_metrics(json_file, output_image)
