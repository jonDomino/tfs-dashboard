"""
Main entry point for processing a game and computing TFS.

Usage:
    from process_game import process_game
    
    tfs_df = process_game(game_id=401700174)
    print(tfs_df)
"""
import pandas as pd
from get_pbp import get_pbp
from preprocess import preprocess_pbp
from compute import compute_tfs


def process_game(game_id: int) -> pd.DataFrame:
    """Process a game and return the final TFS DataFrame.
    
    Args:
        game_id: ESPN game ID (integer)
        
    Returns:
        DataFrame with TFS data (filtered to 3-40 seconds, first field goal attempts only)
        
    Raises:
        ValueError: If no play data found or no valid TFS rows after filtering
    """
    # Fetch raw PBP from ESPN API
    raw_pbp = get_pbp(game_id)
    if raw_pbp is None or len(raw_pbp) == 0:
        raise ValueError(f"No play data found for game {game_id}.")
    
    # Preprocess PBP (clean, add possession context, etc.)
    df = preprocess_pbp(raw_pbp)
    
    # Compute TFS (filter to first field goal attempts, 3-40 seconds)
    tfs_df = compute_tfs(df)
    
    return tfs_df


if __name__ == "__main__":
    # Example usage
    game_id = 401827508
    try:
        tfs_df = process_game(game_id)
        print(f"Successfully processed game {game_id}")
        print(f"Found {len(tfs_df)} valid TFS entries")
        print("\nFirst 10 rows:")
        print(tfs_df[["game_id", "period_number", "possession_id", "action_time", "type_text"]].head(10))
        tfs_df.to_csv(f"outputs/{game_id}.csv", index=False)

        simple_cols = [
            'game_id',
            'sequence_number',
            'possession_id',
            'has_ball_pre_play',
            'has_ball_post_play',
            'poss_start_type',
            'action_time',
            'shot_count_in_poss',
            'is_shot',
            'tfs',
            'poss_start_time',
            'poss_end_time',
            'type_id',
            'type_text',
            'text',
            'away_score',
            'home_score',
            'period_number',
            'clock_value',
            'scoring_play',
            'shooting_play',
            'score_value',
            'modified',
            'team_name',
            'away_team_name',
            'home_team_name',
            'first_ft',
            'final_ft',
            'ft_group_id',
            'chrono_index',

        ]
        simple_df = tfs_df[simple_cols]
        simple_df.to_csv(f"outputs/{game_id}_simple.csv", index=False)
        print(f"Saved to outputs/{game_id}_simple.csv")

    except Exception as e:
        print(f"Error processing game {game_id}: {e}")

