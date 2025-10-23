"""JSON export functionality for benchmark results."""

import json
from pathlib import Path
from typing import Union
from datetime import datetime

from ..metrics import ComprehensiveBenchmarkResults


class JSONExporter:
    """Export benchmark results to JSON format."""
    
    @staticmethod
    def export(results: ComprehensiveBenchmarkResults, output_path: Union[str, Path]) -> None:
        """
        Export benchmark results to a JSON file.
        
        Args:
            results: Comprehensive benchmark results
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to dictionary
        data = results.to_dict()
        
        # Write to file with pretty formatting (UTF-8 for Windows compatibility)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def export_string(results: ComprehensiveBenchmarkResults) -> str:
        """
        Export benchmark results to a JSON string.
        
        Args:
            results: Comprehensive benchmark results
        
        Returns:
            JSON string
        """
        data = results.to_dict()
        return json.dumps(data, indent=2)
    
    @staticmethod
    def load(input_path: Union[str, Path]) -> dict:
        """
        Load benchmark results from a JSON file.
        
        Args:
            input_path: Path to JSON file
        
        Returns:
            Dictionary containing benchmark data
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    @staticmethod
    def export_comparison(results_list: list[ComprehensiveBenchmarkResults], output_path: Union[str, Path]) -> None:
        """
        Export multiple benchmark results for comparison.
        
        Args:
            results_list: List of benchmark results
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        comparison_data = {
            "comparison_timestamp": datetime.now().isoformat(),
            "num_runs": len(results_list),
            "runs": [results.to_dict() for results in results_list]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2)

