import pandas as pd
import re
from typing import Optional
from rapidfuzz import process, fuzz
from sklearn.feature_selection import mutual_info_classif

PATH = "tests/data/Heart_Disease_Prediction.csv"

df = pd.read_csv(PATH)

# normalize column names for fuzzy matching
def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

# perform fuzzy matching on user input to find the best matching column if no exact match is found
def resolve_target_column(df_columns, user_target, threshold=85):
    if not user_target:
        raise ValueError("No target column provided.")

    norm_cols = {_norm(c): c for c in df_columns}
    target_norm = _norm(user_target)

    # exact normalized match
    if target_norm in norm_cols:
        return norm_cols[target_norm]

    # fuzzy match
    matches = process.extract(
        target_norm,
        norm_cols.keys(),
        scorer=fuzz.WRatio,
        limit=5,
    )

    best_norm, best_score, _ = matches[0]
    best_col = norm_cols[best_norm]

    # if the best score is greater than the threshold, return the best column
    if best_score >= threshold:
        print(
            f"Warning: interpreting target '{user_target}' as column '{best_col}' "
            f"(similarity {best_score:.1f})"
        )
        return best_col

    suggestions = ", ".join(
        f"{norm_cols[n]} ({s:.1f})" for n, s, _ in matches
    )
    # if the best score is less than the threshold, raise an error
    raise ValueError(
        f"Could not match target '{user_target}' to any column.\n"
        f"Top candidates: {suggestions}"
    )

# infer the task hint (classification or regression) based on the number of rows and the number of unique target values
def infer_task_hint(n_rows: int, n_unique_target: int, has_target: bool) -> str:
    if not has_target:
        return None

    unique_ratio = n_unique_target / max(n_rows, 1)

    if n_unique_target <= 20 and unique_ratio <= 0.2:
        return "classification"

    return "regression"

# build dataset metadata
def build_dataset_metadata(df, target_column: Optional[str] = None, seed: int = 42) -> dict:
    # collect general info about the dataset
    dataset = {}
    dataset["path"] = PATH # path to the dataset
    dataset["n_rows"] = len(df) # number of rows in the dataset
    dataset["n_cols"] = len(df.columns) # number of columns in the dataset

    total_cells = df.shape[0] * df.shape[1] # total cells in df
    total_nulls = int(df.isna().sum().sum()) # get total null cells
    missing_pct_total = float(total_nulls / max(total_cells, 1)) # get the total missing pct nulls
    dataset["total_nulls"] = total_nulls # total null cells in df
    dataset["missing_pct_total"] = missing_pct_total # pct of total cells missing
    
    dataset["target_column"] = target_column # target column match
    target_present = target_column is not None # whether the target_column is present

    if target_present: # infer task hint (classification or regression) if target is present
        task_hint = infer_task_hint(dataset["n_rows"], df[target_column].nunique(dropna=True), target_present)
    else: # task is just data analysis
        task_hint = "EDA"
    dataset["task_hint"] = task_hint # task hint
    dataset["column_names"] = df.columns.tolist() # list of column names
    dataset["random_seed"] = 42 # random seed for reproducibility
    
    return dataset

# build target summary
def build_target_summary(df, target_column: Optional[str] = None) -> dict:
    target_summary = {}
    
    # get type of target
    n_unique_target = df[target_column].nunique() # n unique labels
    unique_ratio = n_unique_target / max(len(df), 1) # unique labels to rows ratio
    if n_unique_target == 2:
        target_type = "binary"
    elif n_unique_target <= 20 and unique_ratio <= 0.2:
        target_type = "multiclass"
    else:
        target_type = "continuous"
    target_summary["type"] = target_type
    
    # get number of classes
    if n_unique_target <= 20: # check if it's a regression problem first
        target_summary["n_classes"] = n_unique_target
    
    # find imbalance ratio and class distribution
    label_counts = []
    class_distribution = []
    # get label name, count, and percent for each label
    for label in df[target_column].unique().tolist(): 
        count = int((df[target_column] == label).sum())
        label_counts.append(count)
        class_distribution.append({"value": label, "count": count, "pct": round(count/len(df),2)})
    # find imbalance by dividing highest represented label by the lowest
    max_count = max(label_counts)
    min_count = min(label_counts)
    imbalance = max_count / min_count
    
    target_summary["class_imbalance"] = imbalance # set class imbalance
    target_summary["class_distribution"] = class_distribution # show distribution of classes
    
    return target_summary

# create column profiles for the agent
def build_column_profiles(df, target_column: Optional[str] = None):
    column_data = []
    
    # list columns
    columns = df.columns.to_list()
    
    for col in columns:
        column = {}
        column["name"] = col # use column name
        # check if feature or target
        column["role"] = "feature" if col != target_column else "label"
        column["is_target"] = True if col == target_column else False
        
        column["pandas_dtype"] = str(df[col].dtype) # get column dtype
        # find the semantic type for column
        def _infer_semantic_type(s: pd.Series, n_rows: int) -> str:
            if pd.api.types.is_bool_dtype(s):
                return "binary_flag"

            if pd.api.types.is_numeric_dtype(s):
                n_unique = s.nunique(dropna=True)
                unique_ratio = n_unique / max(n_rows, 1)

                if n_unique == 2:
                    return "binary_flag"

                if n_unique <= 20 and unique_ratio <= 0.2:
                    return "categorical"

                return "numeric_continuous"

            return "categorical"
        
        # get semantic type
        semantic_type = _infer_semantic_type(df[col], len(df))
        column["semantic_type"] = semantic_type
        
        # get total null values for column
        n_nulls = int(df[col].isnull().sum()) 
        column["n_missing"] = n_nulls
        column["missing_pct"] = float(n_nulls / max(len(df), 1))
        
        # get n unique values and ratio for column
        n_unique = df[col].nunique(dropna=True)
        column["n_unique"] = n_unique
        column["unique_ratio"] = n_unique / len(df)
        
        # give 5 column value examples
        dropna_col = df[col].dropna()
        column["examples"] = dropna_col.sample(min(5, len(dropna_col)), random_state=42).tolist() # 5 random samples
        
        # get column stats per semantic type
        if semantic_type == "numeric_continuous":
            col_numeric = df[col].describe().to_dict() # get describe nums in dict
            column["numeric_stats"] = {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in col_numeric.items()} # round each value
            
        else: # create stats for a categorical column
            counts = df[col].value_counts(dropna=True)
            column["categorical_stats"] = {"top_values": [
                {"value": str(v), "count": int(c), "pct": round(float(c / counts.sum()),2)}
                for v, c in counts.head(5).items()
            ]}
        # append column to column data list
        column_data.append(column)
        
    return column_data

# extract top numeric correlations
def extract_top_numeric_correlations(corr_matrix, top_k=10):
    pairs = []

    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):  # upper triangle only
            a = cols[i]
            b = cols[j]
            corr = corr_matrix.iloc[i, j]

            pairs.append({
                "col_a": a,
                "col_b": b,
                "corr": round(float(corr), 2),
                "abs_corr": round(float(abs(corr)), 2)
            })

    # sort by absolute strength
    pairs.sort(key=lambda x: x["abs_corr"], reverse=True)

    return pairs[:top_k]

# build numeric target signals
def build_numeric_target_signals(df, column_profiles, target_column, top_k=10):
    signals = []

    # encode target as 0/1
    y = df[target_column].astype("category").cat.codes
    # get numeric features
    for c in column_profiles:
        if c["semantic_type"] == "numeric_continuous" and not c["is_target"]:
            feature = c["name"]
            corr = df[feature].corr(y)
            # if correlation is not NaN, add to signals
            if pd.notna(corr):
                signals.append({
                    "feature": feature,
                    "metric": "point_biserial_corr",
                    "score": round(float(corr), 2),
                    "abs_score": round(abs(float(corr)), 2),
                })

    signals.sort(key=lambda x: x["abs_score"], reverse=True)
    return signals[:top_k]

# ----- helpers to build relationship profiles -----

# build categorical target signals
def build_categorical_target_signals(df, column_profiles, target_column, top_k=10, seed=42):
    # Encode target labels to integers
    y = df[target_column].astype("category").cat.codes

    signals = []

    for c in column_profiles:
        if c["is_target"]:
            continue

        if c["semantic_type"] != "categorical" and c["semantic_type"] != "binary_flag":
            continue

        feature = c["name"]
        x = df[feature]

        # Encode categorical feature to integer codes
        x_codes = x.astype("category").cat.codes

        # mutual_info_classif expects 2D X
        mi = mutual_info_classif(
            X=x_codes.to_frame(),
            y=y,
            discrete_features=True,
            random_state=seed
        )[0]

        signals.append({
            "feature": feature,
            "metric": "mutual_info",
            "score": round(float(mi), 2)
        })

    # Sort strongest first
    signals.sort(key=lambda d: d["score"], reverse=True)
    return signals[:top_k]

# ----- end helpers to build relationship profiles -----

# build relationship profiles
def build_relationship_profiles(df, column_profiles, target_column: Optional[str] = None):
    relationships = {}
    # get numeric columns
    numeric_cols = [
    c["name"] for c in column_profiles
    if c["semantic_type"] == "numeric_continuous"
    ]
    
    # if there are at least 2 numeric columns, get top 10 correlations
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(method="pearson") # get all numeric correlations
        top_pairs = extract_top_numeric_correlations(corr_matrix, top_k=10) # extract top 10 correlations
        # add to relationships for agent
        relationships["numeric_numeric_correlations"] = {
            "method": "pearson",
            "top_abs": top_pairs
        }

    # get numeric target signals
    relationships["numeric_target_signals"] = build_numeric_target_signals(df, column_profiles, target_column, top_k=10)

    # get categorical target signals
    relationships["categorical_target_signals"] = build_categorical_target_signals(df, column_profiles, target_column, top_k=10, seed=42)

    
    return relationships

# grab random rows from dataset to show agent
def build_row_samples(df, n_samples=15, seed=42):
    clean_df = df.dropna()
    if len(clean_df) == 0:
        clean_df = df  # fallback if everything has nulls

    n = min(n_samples, len(clean_df))
    sample_df = clean_df.sample(n=n, random_state=seed)
    return sample_df.to_dict(orient="records")

# combine all the functions to build the complete dataset profile
def build_dataset_profile(df, user_goal, target_column: Optional[str] = None, seed=42):
    profile = {}

    profile["user_goal"] = user_goal # add user's goal first
    profile["dataset"] = build_dataset_metadata(df, target_column) # build dataset metadata
    profile["columns"] = build_column_profiles(df, target_column) # build column profiles
    if target_column: # only build target summary/relationship profiles if target is present
        print("yes target")
        profile["target_summary"] = build_target_summary(df, target_column) # build target summary
        profile["relationships"] = build_relationship_profiles(df, profile["columns"], target_column) # build relationship profiles
        
    profile["row_samples"] = build_row_samples(df) # build row samples

    return profile # return the profile to be shown to the orchestrator agent