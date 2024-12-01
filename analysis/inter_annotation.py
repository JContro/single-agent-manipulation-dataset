import json
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple
import krippendorff
import seaborn as sns
import matplotlib.pyplot as plt

def setup_logging() -> logging.Logger:
    """Configure and return logger with consistent formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def handle_data_files(logger: logging.Logger, download_flag: bool = True) -> Dict[str, Union[dict, pd.DataFrame]]:
    """Handle data files and convert to appropriate format."""
    data_folder = Path('data')
    data_folder.mkdir(exist_ok=True)
    
    files = {
        'manipulation_definitions': 'manipulation-definitions.json',
        'conversations': 'conversations.json',
        'human_responses': 'human_responses.json',
        'user_scores': 'user_scores.json',
        'user_timing': 'user_timing.json'
    }
    
    if download_flag:
        logger.info("Downloading files from GCS bucket")
        from data_connection import create_gcs_file_handler
        file_handler = create_gcs_file_handler('manipulation-dataset-kcl')
        
        for filename in files.values():
            data = file_handler(filename)
            with open(data_folder / filename, 'w') as f:
                json.dump(data, f)
            logger.debug(f"Downloaded and saved {filename}")

    data = {}
    for key, filename in files.items():
        try:
            data[key] = json.load(open(data_folder / filename))
            logger.debug(f"Loaded {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    return data

class ReliabilityAnalyzer:
    """Class to handle inter-rater reliability analysis"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def prepare_annotation_data(self, responses: List[dict]) -> pd.DataFrame:
        """Convert raw responses to structured DataFrame."""
        self.logger.info("Preparing annotation data for reliability analysis")
        
        annotation_df = pd.DataFrame([{
            'rater_id': response['email'],
            'conversation_id': response['conversation_id'],
            'score': response['scores']['General']
        } for response in responses])
        
        self.logger.info(f"Found {len(set(annotation_df.rater_id))} unique raters")
        self.logger.info(f"Found {len(set(annotation_df.conversation_id))} unique conversations")
        
        return annotation_df
    
    def create_reliability_matrix(self, annotation_df: pd.DataFrame) -> np.ndarray:
        """Create matrix for reliability analysis with proper orientation."""
        self.logger.info("Creating reliability matrix")
        
        reliability_matrix = pd.pivot(
            annotation_df,
            index='rater_id',
            columns='conversation_id',
            values='score'
        ).to_numpy()
        
        self.logger.info(f"Created matrix with shape {reliability_matrix.shape}")
        return reliability_matrix
    
    def transform_to_trinary(self, matrix: np.ndarray) -> np.ndarray:
        """Transform scores to trinary scale (-1, 0, 1)."""
        self.logger.info("Transforming scores to trinary scale")
        
        transformed = matrix.copy()
        transformed = np.where(transformed < 4, -1, transformed)
        transformed = np.where(transformed == 4, 0, transformed)
        transformed = np.where(transformed > 4, 1, transformed)
        
        unique_values = np.unique(transformed[~np.isnan(transformed)])
        self.logger.info(f"Transformed values present: {unique_values}")
        
        return transformed
    
    def calculate_reliability(self, matrix: np.ndarray, 
                            level: str = 'ordinal') -> float:
        """Calculate Krippendorff's alpha for given matrix."""
        self.logger.info(f"Calculating Krippendorff's alpha with {level} level")
        
        alpha = krippendorff.alpha(reliability_data=matrix, 
                                 level_of_measurement=level)
        
        self.logger.info(f"Krippendorff's alpha: {alpha:.3f}")
        self._log_reliability_interpretation(alpha)
        
        return alpha

    def analyze_rater_subgroups(self, annotation_df: pd.DataFrame, min_ratings: int = 10) -> Dict:
        """Analyze reliability for raters with minimum number of ratings."""
        self.logger.info(f"Analyzing rater subgroups with minimum {min_ratings} ratings")
        
        # Get counts of ratings per rater
        rating_counts = annotation_df.groupby('rater_id').size()
        frequent_raters = rating_counts[rating_counts >= min_ratings]
        
        # Filter for frequent raters
        filtered_df = annotation_df[annotation_df['rater_id'].isin(frequent_raters.index)]
        
        self.logger.info(f"Found {len(frequent_raters)} raters with {min_ratings}+ ratings")
        self.logger.info(f"Average ratings per frequent rater: {rating_counts[frequent_raters.index].mean():.1f}")
        
        # Calculate reliability for frequent raters
        reliability_matrix = self.create_reliability_matrix(filtered_df)
        alpha = self.calculate_reliability(reliability_matrix)
        
        return {
            'frequent_raters': frequent_raters,
            'filtered_df': filtered_df,
            'alpha': alpha
        }

    def analyze_pairwise_agreement(self, annotation_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate agreement between pairs of raters who rated the same items."""
        self.logger.info("Analyzing pairwise agreement between raters")
        
        pairs = []
        for conv_id in annotation_df['conversation_id'].unique():
            conv_ratings = annotation_df[annotation_df['conversation_id'] == conv_id]
            if len(conv_ratings) >= 2:
                for i, row1 in conv_ratings.iterrows():
                    for j, row2 in conv_ratings.iterrows():
                        if i < j:
                            pairs.append({
                                'rater1': row1['rater_id'],
                                'rater2': row2['rater_id'],
                                'conversation_id': conv_id,
                                'score1': row1['score'],
                                'score2': row2['score'],
                                'agreement': int(row1['score'] == row2['score']),
                                'diff': abs(row1['score'] - row2['score'])
                            })
        
        pairs_df = pd.DataFrame(pairs)
        
        self.logger.info(f"Analyzed {len(pairs_df)} rating pairs")
        self.logger.info(f"Average agreement rate: {pairs_df['agreement'].mean():.2%}")
        self.logger.info(f"Average score difference: {pairs_df['diff'].mean():.2f}")
        
        return pairs_df

    def analyze_score_distribution(self, annotation_df: pd.DataFrame) -> Dict:
        """Analyze the distribution of scores across raters."""
        self.logger.info("Analyzing score distribution")
        
        overall_dist = annotation_df['score'].value_counts(normalize=True).sort_index()
        rater_dist = annotation_df.groupby('rater_id')['score'].value_counts(normalize=True).unstack()
        
        rater_stats = {
            'mean': annotation_df.groupby('rater_id')['score'].mean(),
            'std': annotation_df.groupby('rater_id')['score'].std(),
            'count': annotation_df.groupby('rater_id')['score'].count()
        }
        
        self.logger.info("\nOverall score distribution:")
        for score, freq in overall_dist.items():
            self.logger.info(f"Score {score}: {freq:.1%}")
        
        self.logger.info(f"\nRater statistics:")
        self.logger.info(f"Mean score range: {rater_stats['mean'].min():.2f} - {rater_stats['mean'].max():.2f}")
        self.logger.info(f"Standard deviation range: {rater_stats['std'].min():.2f} - {rater_stats['std'].max():.2f}")
        
        # Create plots directory if it doesn't exist
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        
        # Plot overall distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=annotation_df, x='score', bins=5)
        plt.title('Overall Score Distribution')
        plt.savefig(plots_dir / 'score_distribution.png')
        plt.close()
        
        # Plot rater means distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=rater_stats['mean'], bins=20)
        plt.title('Distribution of Rater Mean Scores')
        plt.savefig(plots_dir / 'rater_means_distribution.png')
        plt.close()
        
        return {
            'overall_distribution': overall_dist,
            'rater_distribution': rater_dist,
            'rater_stats': rater_stats
        }
    
    def _log_reliability_interpretation(self, alpha: float):
        """Log interpretation of reliability coefficient."""
        if alpha >= 0.800:
            self.logger.info("Reliability level: Strong agreement")
        elif alpha >= 0.667:
            self.logger.info("Reliability level: Moderate agreement")
        elif alpha >= 0:
            self.logger.info("Reliability level: Weak agreement")
        else:
            self.logger.info("Reliability level: Systematic disagreement")

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting reliability analysis pipeline")
    
    # Load data
    data = handle_data_files(logger, download_flag=False)
    
    # Process responses
    from utils.filtering_testing import remove_bad_responses
    
    initial_responses = len(data['human_responses'])
    clean_responses = remove_bad_responses(
        human_responses=data['human_responses'], 
        user_timing=data['user_timing']
    )
    final_responses = len(clean_responses)
    
    logger.info("Response cleaning results:")
    logger.info(f"Initial responses: {initial_responses}")
    logger.info(f"Valid responses: {final_responses}")
    logger.info(f"Removed responses: {initial_responses - final_responses}")
    logger.info(f"Removal rate: {((initial_responses - final_responses) / initial_responses * 100):.2f}%")
    
    # Initialize reliability analyzer
    analyzer = ReliabilityAnalyzer(logger)
    
    # Prepare data and calculate reliability
    annotation_df = analyzer.prepare_annotation_data(clean_responses)
    reliability_matrix = analyzer.create_reliability_matrix(annotation_df)
    
    # Calculate original reliability
    original_alpha = analyzer.calculate_reliability(reliability_matrix)
    
    # Calculate trinary reliability
    trinary_matrix = analyzer.transform_to_trinary(reliability_matrix)
    trinary_alpha = analyzer.calculate_reliability(trinary_matrix)
    
    # Additional analyses
    frequent_rater_analysis = analyzer.analyze_rater_subgroups(annotation_df, min_ratings=10)
    pairwise_agreement = analyzer.analyze_pairwise_agreement(annotation_df)
    score_distribution = analyzer.analyze_score_distribution(annotation_df)
    
    logger.info("Reliability analysis completed")
    
    return {
        'original_alpha': original_alpha,
        'trinary_alpha': trinary_alpha,
        'annotation_df': annotation_df,
        'reliability_matrix': reliability_matrix,
        'trinary_matrix': trinary_matrix,
        'frequent_rater_analysis': frequent_rater_analysis,
        'pairwise_agreement': pairwise_agreement,
        'score_distribution': score_distribution
    }

if __name__ == "__main__":
    results = main()