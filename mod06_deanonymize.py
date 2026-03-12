import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    merged = pd.merge(anon_df, aux_df, on=['age', 'gender', 'zip3'])

    counts = merged['anon_id'].value_counts()
    unique_anon_ids = counts[counts == 1].index

    matches = merged[merged['anon_id'].isin(unique_anon_ids)]

    return matches[['anon_id', 'name']].rename(columns={'name': 'matched_name'})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    
    return len(matches_df)/len(anon_df)
