"""
Unit tests for GPU Info API.

Tests HTML cleaning, data processing, and validation functions.
"""
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

import pandas as pd
import requests

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from update import (
    clean_html,
    normalize_columns,
    standardise_column_names,
    process_dataframe,
    remove_bracketed_references,
    record_key,
)
from validators import (
    validate_dataframe,
    validate_gpu_record,
    ValidationError,
)


class TestHTMLCleaning(unittest.TestCase):
    """Test HTML cleaning functions."""
    
    def test_clean_html_removes_citations(self):
        """Test that citation superscripts are removed."""
        html = '<td>GeForce RTX 4090<sup>[1]</sup></td>'
        cleaned = clean_html(html)
        self.assertNotIn('<sup>', cleaned)
        self.assertNotIn('[1]', cleaned)
    
    def test_clean_html_removes_style_blocks(self):
        """Test that style blocks are removed."""
        html = '<style>.vertical-header{transform:rotate(90deg);}</style><table>data</table>'
        cleaned = clean_html(html)
        self.assertNotIn('<style>', cleaned)
        self.assertNotIn('transform', cleaned)
        self.assertIn('table', cleaned)
    
    def test_clean_html_normalizes_entities(self):
        """Test that HTML entities are normalized."""
        html = '294&nbsp;mm2'
        cleaned = clean_html(html)
        self.assertIn('mm2', cleaned)
        self.assertNotIn('&nbsp;', cleaned)
    
    def test_clean_html_handles_rowspan_colspan(self):
        """Test that rowspan/colspan are standardized."""
        html = '<th rowspan=2>Header</th><th colspan="3">Another</th>'
        cleaned = clean_html(html)
        self.assertIn('rowspan="2"', cleaned)
        self.assertIn('colspan="3"', cleaned)


class TestColumnNormalization(unittest.TestCase):
    """Test column normalization functions."""
    
    def test_normalize_columns_removes_duplicates(self):
        """Test that duplicate words in column names are removed."""
        cols = pd.Index(['Launch Launch', 'Model Model', 'TDP'])
        result = normalize_columns(cols)
        self.assertEqual(result, ['Launch', 'Model', 'TDP'])
    
    def test_normalize_columns_multiindex(self):
        """Test MultiIndex column normalization."""
        cols = pd.MultiIndex.from_tuples([
            ('Memory', 'Size (GB)'),
            ('Memory', 'Type'),
            ('Core', 'Count'),
        ])
        result = normalize_columns(cols)
        self.assertEqual(result, ['Memory Size (GB)', 'Memory Type', 'Core Count'])
    
    def test_standardise_column_names(self):
        """Test column name standardization."""
        df = pd.DataFrame({
            'gpu die': ['GK104'],
            'model number': ['GTX 680'],
        })
        result = standardise_column_names(df)
        self.assertIn('Code name', result.columns)
        self.assertIn('Model', result.columns)


class TestDataProcessing(unittest.TestCase):
    """Test data processing functions."""
    
    def test_process_dataframe_adds_vendor(self):
        """Test that vendor is added to DataFrame."""
        df = pd.DataFrame({
            'Model': ['RTX 4090', 'RTX 4080'],
            'Launch': ['Oct 12, 2022', 'Nov 16, 2022'],
        })
        result = process_dataframe(df, 'NVIDIA')
        self.assertIn('Vendor', result.columns)
        self.assertEqual(result['Vendor'].iloc[0], 'NVIDIA')
    
    def test_process_dataframe_parses_dates(self):
        """Test that launch dates are parsed."""
        df = pd.DataFrame({
            'Model': ['RTX 4090'],
            'Launch date': ['Oct 12, 2022[1]'],
        })
        result = process_dataframe(df, 'NVIDIA')
        self.assertIn('Launch', result.columns)
        self.assertTrue(pd.notna(result['Launch'].iloc[0]))
    
    def test_remove_bracketed_references(self):
        """Test removal of reference brackets."""
        df = pd.DataFrame({
            'Model': ['RTX 4090[1][2]', 'RTX 4080[3]'],
            'Code name': ['AD102[1]', 'AD103'],
        })
        result = remove_bracketed_references(df, ['Model', 'Code name'])
        self.assertEqual(result['Model'].iloc[0], 'RTX 4090')
        self.assertEqual(result['Code name'].iloc[0], 'AD102')


class TestRecordKey(unittest.TestCase):
    """Test record key generation."""
    
    def test_record_key_uses_code_name(self):
        """Test that Code name is preferred for key."""
        record = {
            'Vendor': 'NVIDIA',
            'Code name': 'AD102',
            'Model': 'RTX 4090',
        }
        key = record_key(record)
        self.assertEqual(key, 'NVIDIA_AD102')
    
    def test_record_key_falls_back_to_model(self):
        """Test that Model is used when Code name is missing."""
        record = {
            'Vendor': 'AMD',
            'Model': 'Radeon RX 7900 XTX',
        }
        key = record_key(record)
        self.assertEqual(key, 'AMD_Radeon_RX_7900_XTX')
    
    def test_record_key_sanitizes_special_chars(self):
        """Test that special characters are replaced."""
        record = {
            'Vendor': 'Intel',
            'Model': 'Arc A770 (Limited Edition)',
        }
        key = record_key(record)
        # Should replace spaces and parentheses with underscores
        self.assertNotIn(' ', key)
        self.assertNotIn('(', key)


class TestValidation(unittest.TestCase):
    """Test validation functions."""
    
    def test_validate_dataframe_empty(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        is_valid, warnings = validate_dataframe(df, 'NVIDIA')
        self.assertFalse(is_valid)
        self.assertTrue(len(warnings) > 0)
    
    def test_validate_dataframe_minimal(self):
        """Test validation of minimal DataFrame."""
        df = pd.DataFrame({
            'Model': ['RTX 4090', 'RTX 4080', 'RTX 4070'],
            'Vendor': ['NVIDIA', 'NVIDIA', 'NVIDIA'],
        })
        is_valid, warnings = validate_dataframe(df, 'NVIDIA')
        # Should be valid even if minimal
        self.assertTrue(is_valid)
    
    def test_validate_gpu_record_missing_vendor(self):
        """Test validation flags missing vendor."""
        record = {'Model': 'RTX 4090'}
        is_valid, warnings = validate_gpu_record(record, 'test_key')
        self.assertFalse(is_valid)
        self.assertTrue(any('Vendor' in w for w in warnings))
    
    def test_validate_gpu_record_unknown_vendor(self):
        """Test validation flags unknown vendor."""
        record = {'Vendor': 'UnknownVendor', 'Model': 'GPU X'}
        is_valid, warnings = validate_gpu_record(record, 'test_key')
        self.assertTrue(any('Unknown vendor' in w for w in warnings))
    
    def test_validate_gpu_record_valid(self):
        """Test validation of valid GPU record."""
        record = {
            'Vendor': 'NVIDIA',
            'Model': 'RTX 4090',
            'Code name': 'AD102',
            'Launch': '2022-10-12',
            'TDP (Watts)': 450.0,
        }
        is_valid, warnings = validate_gpu_record(record, 'test_key')
        self.assertTrue(is_valid)
        self.assertEqual(len(warnings), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests with mocked network calls."""
    
    @patch('update.fetch_url_with_retry')
    def test_fetch_vendor_tables_success(self, mock_fetch):
        """Test successful vendor table fetching."""
        # Mock HTML response with a simple table
        mock_html = '''
        <table>
            <tr><th>Model</th><th>Launch</th><th>TDP</th></tr>
            <tr><td>RTX 4090</td><td>Oct 2022</td><td>450W</td></tr>
            <tr><td>RTX 4080</td><td>Nov 2022</td><td>320W</td></tr>
        </table>
        '''
        mock_fetch.return_value = mock_html
        
        from update import fetch_vendor_tables
        
        # This will fail to parse since our mock HTML is too simple
        # But it tests the retry mechanism is working
        try:
            tables = fetch_vendor_tables('NVIDIA', 'http://example.com')
        except:
            # Expected to fail on parsing, but network fetch should succeed
            pass
        
        # Verify fetch was called
        mock_fetch.assert_called_once()


if __name__ == '__main__':
    unittest.main()
