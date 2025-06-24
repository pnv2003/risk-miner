def preprocess_data(data, compact=False):
    # this is applied for training and testing data
    # here we perform the preprocessing steps, derived from EDA / feature engineering (from train only)

    df = data.copy()

    df['open_issues'] = df['num_status_to_do'] + df['num_status_in_progress']
    df['incomplete_ratio'] = df['open_issues'] / df['total_issues']

    df['reopen_ratio'] = df['total_reopened'] / df['total_issues']
    df["fix_rate"] = df["num_resolution_fixed"] / df["total_issues"]
    df['assignee_reporter_ratio'] = df['total_assignees'] / (df['total_reporters'] + 1e-5)
    df['num_issue_type_bug_ratio'] = df['num_issue_type_bug'] / df['total_issues']
    df['high_priority_bug_ratio'] = df['num_high_priority_bugs'] / df['total_issues']

    status_cols = ['num_status_to_do', 'num_status_in_progress', 'num_status_done', 'num_status_other']
    # Calculate percentages instead of absolute counts
    for col in status_cols:
        df[f"{col}_pct"] = (df[col] / df['total_issues']) * 100

    resolution_cols = [
        'num_resolution_fixed', 'num_resolution_duplicate', 'num_resolution_won\'t_fix',
        'num_resolution_invalid', 'num_resolution_incomplete', 'num_resolution_other'
    ]
    # Calculate percentages
    for col in resolution_cols:
        if col in df.columns:
            df[f"{col}_pct"] = (df[col] / df['total_issues']) * 100

    # Analyze issue types
    type_cols = ['num_issue_type_bug', 'num_issue_type_feature', 'num_issue_type_improvement',
                'num_issue_type_task', 'num_issue_type_other']
    type_cols = [col for col in type_cols if col in df.columns]

    # Calculate proportions
    for col in type_cols:
        df[f"{col}_ratio"] = df[col] / df['total_issues']

    priority_cols = ['num_priority_high', 'num_priority_medium', 'num_priority_low', 'num_priority_other']
    priority_cols = [col for col in priority_cols if col in df.columns]

    # Calculate proportions
    for col in priority_cols:
        df[f"{col}_ratio"] = df[col] / df['total_issues']

    development_cols = ['num_issue_type_feature', 'num_issue_type_improvement', 'num_issue_type_task']
    available_dev_cols = [col for col in development_cols if col in df.columns]

    # Sum all development work issues
    df['total_development_issues'] = df[available_dev_cols].sum(axis=1)
    # Calculate the ratio
    df['bug_to_development_ratio'] = df['num_issue_type_bug'] / (df['total_development_issues'] + 1e-5)

    resolution_cols = [
        'num_resolution_fixed', 'num_resolution_duplicate', 'num_resolution_won\'t_fix',
        'num_resolution_invalid', 'num_resolution_incomplete', 'num_resolution_other'
    ]
    df["resolved_issues"] = df[resolution_cols].sum(axis=1)
    df["resolution_ratio"] = df["resolved_issues"] / df["total_issues"]

    median_lifespan = df['average_lifespan'].median()
    df['average_lifespan'] = df['average_lifespan'].fillna(median_lifespan * 2) # Assuming stalled projects have double the median lifespan

    # drop unused columns (derived from EDA)
    eda_unused_columns = [
        col for col in df.columns 
        if col.startswith('log1p_') or
        (col.startswith('num_') and ('pct' not in col and 'ratio' not in col)) or # exclude raw counts
        col in [
            'volatility_score', 
            'total_development_issues', 
            'size_quartile', 
            'open_issue_ratio', # duplicated by incomplete_ratio
            'fix_rate', # duplicated by num_resolution_fixed_pct
            'assignee_reporter_group',
            'open_issues',
            'total_reopened', # we use reopen_ratio instead
            'resolved_issues', # we use resolution_ratio instead
        ]
    ]

    if compact: # if we want to avoid multicollinearity
        # drop unused columns (derived from feature engineering)
        fe_unused_columns = [
            # 'resolution_ratio', # high VIF score
            'num_status_done_pct', # high VIF score
            'num_status_to_do_pct', # high VIF score
            'num_resolution_other_pct', # high VIF score
            'num_issue_type_task_ratio', # high VIF score
            'num_issue_type_improvement_ratio', # high VIF score
            'num_issue_type_feature_ratio', # high VIF score
            # 'high_priority_bug_ratio', # highly correlated with num_high_priority_ratio
        ]
    else:
        fe_unused_columns = []

    # drop unused columns (derived from labeling)
    labeling_unused_columns = [
        col for col in df.columns
        if 'score' in col or 'majority_vote' in col
    ]

    # Drop unused columns
    df.drop(columns=eda_unused_columns + fe_unused_columns + labeling_unused_columns, inplace=True, errors='ignore')

    return df
