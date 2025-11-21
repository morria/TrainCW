"""Command-line interface for TrainCW."""

import argparse
import sys


def main() -> int:
    """Main entry point for the TrainCW CLI.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        prog="traincw",
        description="Neural network training system for Morse code decoding",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration file",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to Core ML")
    export_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    export_parser.add_argument(
        "--output",
        type=str,
        help="Output path for exported model",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Command implementations will be added as modules are developed
    print(f"Command '{args.command}' will be implemented in future modules.")
    print("The project structure is now set up and ready for development!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
