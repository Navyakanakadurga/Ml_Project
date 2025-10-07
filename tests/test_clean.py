from src.data.clean import standardize
import pandas as pd
def test_standardize_minimal():
    df = pd.DataFrame({'date':['2020-01-01','2020-01-02'],'amount':[10,20]})
    out = standardize(df)
    assert 'date' in out.columns and 'amount' in out.columns
