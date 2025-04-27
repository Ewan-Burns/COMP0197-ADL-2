import argparse

from src.ablation_study.ablation_trainer import run_ablation_study
from src.ablation_study.ablation_visualizer import visualize_ablation_results, load_and_visualize_results


def main():
    """Main function to run the ablation study or visualize existing results."""
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='Run ablation study on SEC model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--visualize-only', action='store_true', help='Only visualize existing results')
    parser.add_argument('--results-file', type=str, default='./results/ablation_results.csv', 
                        help='Path to results CSV file (for --visualize-only)')
    
    args = parser.parse_args()
    
    if args.visualize_only:
        try:
            print(f"Loading existing results from {args.results_file} for visualization...")
            load_and_visualize_results(args.results_file)
        except FileNotFoundError:
            print(f"Results file {args.results_file} not found. Running full ablation study instead.")
            # Run the full ablation study
            results, histories = run_ablation_study(
                num_epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size
            )
            
            # Visualize the results
            visualize_ablation_results(results, histories)
    else:
        print(f"Running ablation study with {args.epochs} epochs, learning rate {args.lr}, batch size {args.batch_size}...")
        # Run the full ablation study
        results, histories = run_ablation_study(
            num_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )
        
        # Visualize the results
        visualize_ablation_results(results, histories)


if __name__ == "__main__":
    main()