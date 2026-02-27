import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from cosmeticos_ia.features.geocode import geocode_clientes


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "endereco": [
            "Rua A, 123",
            "Rua B, 456",
            "Rua C, 789",
        ]
    })


def test_geocode_clientes_success(sample_df):
    """
    Test geocoding of clients successfully.
    """
    with patch("cosmeticos_ia.features.geocode.Nominatim") as mock_nominatim:
        # Create a mock geolocator that returns a mock location
        mock_geolocator = MagicMock()
        mock_location = MagicMock()
        mock_location.latitude = -8.05428
        mock_location.longitude = -34.8813
        mock_geolocator.geocode.return_value = mock_location
        mock_nominatim.return_value = mock_geolocator

        # Call the function
        result_df = geocode_clientes(sample_df, cache_path=None)

        # Assertions
        assert "lat" in result_df.columns
        assert "lon" in result_df.columns
        assert not result_df["lat"].isnull().any()
        assert not result_df["lon"].isnull().any()


def test_geocode_clientes_empty_df():
    """
    Test geocoding with an empty DataFrame.
    """
    # Create an empty DataFrame
    empty_df = pd.DataFrame({"endereco": []})

    # Call the function
    result_df = geocode_clientes(empty_df, cache_path=None)

    # Assertions
    assert "lat" in result_df.columns
    assert "lon" in result_df.columns
    assert result_df.empty


def test_geocode_clientes_missing_column(sample_df):
    """
    Test geocoding with a missing address column.
    """
    # Create a DataFrame without the 'endereco' column
    df_missing_col = sample_df.drop(columns=["endereco"])

    # Call the function and expect a ValueError
    with pytest.raises(ValueError):
        geocode_clientes(df_missing_col, cache_path=None)
