import pytest
from clairvoyants import aggregation
import pandas as pd

# Testing the aggregation functionality

def test_aggregate_to_longest():
    data = pd.DataFrame({'dt': pd.date_range('2020-01-01',
     periods=14, freq='D'), 'actual': [1, 1, 1, 1, 1, 1, 7, 2, 2,
     2, 2, 2, 2, 14]})
    result = aggregation.aggregate_to_longest(data)
    assert all(result['period7'].actual == [13, 26])
    assert all(result['period7'].dt == ['2020-01-07', 
      '2020-01-14'])
