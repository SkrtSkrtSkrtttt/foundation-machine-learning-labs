# ESE 388 - Lab 2: Python Data Structures
# Author: Naafiul Hossain
# Date: 9/3/25

import numpy as np
import pandas as pd


def make_dataset(seed: int | None = 42) -> pd.DataFrame:
    """
    Create a DataFrame with:
      - Student ID: 1..20
      - Hours Studied: random ints [1, 10]
      - Exam Score: random ints [50, 100]
    A seed is used for reproducibility unless set to None.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    student_id = np.arange(1, 21, dtype=int)
    hours_studied = rng.integers(low=1, high=11, size=20)        # 1..10 inclusive
    exam_score = rng.integers(low=50, high=101, size=20)         # 50..100 inclusive

    df = pd.DataFrame({
        "Student ID": student_id,
        "Hours Studied": hours_studied,
        "Exam Score": exam_score,
    })
    return df


# --------------------------
# Required functions
# --------------------------

def scores_above_70(scores_list: list[int]) -> list[int]:
    """
    List Function:
    Given a list of exam scores, return a list of all scores > 70.
    """
    return [s for s in scores_list if s > 70]


def min_max_from_tuple(scores_tuple: tuple[int, ...]) -> tuple[int, int]:
    """
    Tuple Function:
    Given a tuple of exam scores, return (min_score, max_score).
    """
    if not scores_tuple:
        raise ValueError("scores_tuple is empty")
    return (min(scores_tuple), max(scores_tuple))


def id_to_score_map(df: pd.DataFrame) -> dict[int, int]:
    """
    Dictionary Function:
    Given the DataFrame, return a dict mapping Student ID -> Exam Score.
    """
    # Validate required columns exist
    required = {"Student ID", "Exam Score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    return dict(zip(df["Student ID"].astype(int), df["Exam Score"].astype(int), strict=True))


# --------------------------
# random main genrator
# --------------------------

if __name__ == "__main__":
    # 1) Create the dataset
    df = make_dataset(seed=42)

    print("=== Original DataFrame ===")
    print(df.to_string(index=False))

    # 2) Prepare inputs for each function
    exam_scores_list = df["Exam Score"].tolist()
    exam_scores_tuple = tuple(df["Exam Score"].tolist())

    # 3) Test each function and print results
    print("\n=== List Function: scores > 70 ===")
    above_70 = scores_above_70(exam_scores_list)
    print(above_70)

    print("\n=== Tuple Function: (min, max) ===")
    min_score, max_score = min_max_from_tuple(exam_scores_tuple)
    print(f"min={min_score}, max={max_score}")

    print("\n=== Dictionary Function: {Student ID: Exam Score} ===")
    mapping = id_to_score_map(df)
    print(mapping)

    # 4) Summary statistics and sorted data
    print("\n=== Summary Statistics (describe) ===")
    print(df.describe())
    # naafiul note-u might need to downgrade pandas

    print("\n=== Sorted by Exam Score (descending) ===")
    sorted_df = df.sort_values(by="Exam Score", ascending=False, kind="mergesort")
    print(sorted_df.to_string(index=False))
