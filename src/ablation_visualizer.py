import matplotlib.pyplot as plt
import numpy as np


def visualize_ablation_results(results, training_histories=None):
    """Visualize the results of the ablation study."""
    
    # 1. Bar chart comparing mIoU and mDice for different ablations
    names = list(results.keys())
    mious = [results[name]['miou'] for name in names]
    mdices = [results[name]['mdice'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    try:
        # Use QtAgg backend if available (consistent with rest of code)
        import matplotlib
        matplotlib.use("QtAgg")
    except:
        # Fall back to a default backend if QtAgg is not available
        pass
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, mious, width, label='mIoU')
    rects2 = ax.bar(x + width/2, mdices, width, label='mDice')
    
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study Results')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('./results/ablation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. If training histories are available, plot learning curves
    if training_histories:
        # Plot training loss
        plt.figure(figsize=(12, 6))
        for name, history in training_histories.items():
            plt.plot(history['train_loss'], label=f"{name} (Train)")
        plt.title('Training Loss by Ablation Configuration')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('./results/ablation_train_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot validation loss
        plt.figure(figsize=(12, 6))
        for name, history in training_histories.items():
            plt.plot(history['val_loss'], label=f"{name} (Val)")
        plt.title('Validation Loss by Ablation Configuration')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('./results/ablation_val_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Plot component weights vs. performance
    alphas = [results[name]['alpha'] for name in names]
    betas = [results[name]['beta'] for name in names]
    gammas = [results[name]['gamma'] for name in names]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Alpha (Seed) vs. Performance
    ax1.scatter(alphas, mious, label='mIoU', color='blue')
    ax1.scatter(alphas, mdices, label='mDice', color='orange')
    for i, name in enumerate(names):
        ax1.annotate(name, (alphas[i], mious[i]), textcoords="offset points", 
                     xytext=(0,10), ha='center')
    ax1.set_xlabel('Alpha (Seed)')
    ax1.set_ylabel('Score')
    ax1.set_title('Impact of Seed Loss Weight')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Beta (Expand) vs. Performance
    ax2.scatter(betas, mious, label='mIoU', color='blue')
    ax2.scatter(betas, mdices, label='mDice', color='orange')
    for i, name in enumerate(names):
        ax2.annotate(name, (betas[i], mious[i]), textcoords="offset points", 
                     xytext=(0,10), ha='center')
    ax2.set_xlabel('Beta (Expand)')
    ax2.set_ylabel('Score')
    ax2.set_title('Impact of Expand Loss Weight')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Gamma (Constrain) vs. Performance
    ax3.scatter(gammas, mious, label='mIoU', color='blue')
    ax3.scatter(gammas, mdices, label='mDice', color='orange')
    for i, name in enumerate(names):
        ax3.annotate(name, (gammas[i], mious[i]), textcoords="offset points", 
                     xytext=(0,10), ha='center')
    ax3.set_xlabel('Gamma (Constrain)')
    ax3.set_ylabel('Score')
    ax3.set_title('Impact of Constrain Loss Weight')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('./results/ablation_component_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization complete. Results saved to ./results/ directory.")


def load_and_visualize_results(results_file='./results/ablation_results.txt'):
    """Load results from a text file and visualize them."""
    # Load results from simple text file format
    results = {}
    current_config = None
    
    with open(results_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if ':' in line and 'alpha:' not in line and 'beta:' not in line:
                # This is a config name
                current_config = line.replace(':', '')
                results[current_config] = {}
            elif current_config and ':' in line:
                # This is a property
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    # Try to convert to float for numeric values
                    results[current_config][key] = float(value)
                except ValueError:
                    # Keep as string for non-numeric values
                    results[current_config][key] = value
    
    # Visualize the results
    visualize_ablation_results(results)