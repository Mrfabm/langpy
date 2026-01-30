"""
Advanced Filter Expression Parser - Langbase-style filter DSL implementation.

Supports advanced filter expressions like:
- source = "file.pdf"
- custom.author = "Alice"
- timestamp > "2024-01-01"
- AND/OR compound filters
- Nested filter expressions
"""

import re
import json
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

from .models import FilterExpression, CompoundFilter, FilterType


class AdvancedFilterParser:
    """Advanced filter expression parser for memory queries (Langbase-style)."""
    
    # Supported operators mapping
    OPERATORS = {
        'eq': 'eq',
        'neq': 'neq', 
        'in': 'in',
        'nin': 'nin',
        'gt': 'gt',
        'gte': 'gte',
        'lt': 'lt',
        'lte': 'lte',
        'contains': 'contains'
    }
    
    def __init__(self):
        """Initialize the advanced filter parser."""
        # Regex pattern for parsing simple filter expressions
        self.simple_pattern = re.compile(
            r'(\w+(?:\.\w+)*)\s*([=!<>]+|in|contains)\s*["\']([^"\']*)["\']'
        )
    
    def parse(self, filter_input: Union[str, Dict, List]) -> FilterType:
        """
        Parse a filter input into a FilterType (Langbase-style).
        
        Args:
            filter_input: Filter input (string, dict, or list)
            
        Returns:
            Parsed filter expression
            
        Examples:
            >>> parser = AdvancedFilterParser()
            >>> parser.parse('source = "file.pdf"')
            FilterExpression(field="source", operator="eq", value="file.pdf")
            
            >>> parser.parse({
            ...     "and": [
            ...         {"field": "team", "operator": "eq", "value": "sales"},
            ...         {"field": "region", "operator": "in", "value": ["EU", "APAC"]}
            ...     ]
            ... })
            CompoundFilter(and_conditions=[...])
        """
        if isinstance(filter_input, str):
            return self._parse_string(filter_input)
        elif isinstance(filter_input, dict):
            return self._parse_dict(filter_input)
        elif isinstance(filter_input, list):
            # Convert list to AND condition
            return CompoundFilter(and_conditions=[
                self._parse_dict(item) if isinstance(item, dict) else item
                for item in filter_input
            ])
        else:
            raise ValueError(f"Unsupported filter input type: {type(filter_input)}")
    
    def _parse_string(self, filter_expr: str) -> FilterType:
        """Parse a string filter expression."""
        if not filter_expr or not filter_expr.strip():
            return {}
        
        # Check if it's a JSON-like structure
        if filter_expr.strip().startswith('{'):
            try:
                return self._parse_dict(json.loads(filter_expr))
            except json.JSONDecodeError:
                pass
        
        # Parse as simple expression
        return self._parse_simple_expression(filter_expr)
    
    def _parse_dict(self, filter_dict: Dict[str, Any]) -> FilterType:
        """Parse a dictionary filter expression."""
        # Check for compound filters
        if "and" in filter_dict or "or" in filter_dict:
            return self._parse_compound_filter(filter_dict)
        
        # Check for individual filter expression
        if "field" in filter_dict and "operator" in filter_dict:
            return self._parse_filter_expression(filter_dict)
        
        # Fallback to simple key-value dict
        return filter_dict
    
    def _parse_compound_filter(self, filter_dict: Dict[str, Any]) -> CompoundFilter:
        """Parse a compound filter with AND/OR logic."""
        and_conditions = None
        or_conditions = None
        
        if "and" in filter_dict:
            and_conditions = [
                self._parse_dict(item) if isinstance(item, dict) else item
                for item in filter_dict["and"]
            ]
        
        if "or" in filter_dict:
            or_conditions = [
                self._parse_dict(item) if isinstance(item, dict) else item
                for item in filter_dict["or"]
            ]
        
        return CompoundFilter(
            and_conditions=and_conditions,
            or_conditions=or_conditions
        )
    
    def _parse_filter_expression(self, filter_dict: Dict[str, Any]) -> FilterExpression:
        """Parse an individual filter expression."""
        return FilterExpression(
            field=filter_dict["field"],
            operator=filter_dict["operator"],
            value=self._parse_value(filter_dict["value"], filter_dict["operator"])
        )
    
    def _parse_simple_expression(self, filter_expr: str) -> Dict[str, Any]:
        """Parse a simple filter expression string."""
        filters = {}
        
        # Find all matches in the expression
        matches = self.simple_pattern.findall(filter_expr)
        
        for field, operator, value in matches:
            # Handle nested fields (e.g., custom.author)
            if '.' in field:
                parts = field.split('.')
                if parts[0] == 'custom':
                    if 'custom' not in filters:
                        filters['custom'] = {}
                    filters['custom'][parts[1]] = self._parse_value(value, operator)
                else:
                    # For other nested fields, flatten them
                    filters[field] = self._parse_value(value, operator)
            else:
                filters[field] = self._parse_value(value, operator)
        
        return filters
    
    def _parse_value(self, value: str, operator: str) -> Any:
        """
        Parse a value based on the operator.
        
        Args:
            value: The value string
            operator: The comparison operator
            
        Returns:
            Parsed value
        """
        # Try to parse as datetime if it looks like a date
        if self._looks_like_date(value):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                pass
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if a value looks like a date string."""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',  # ISO with Z
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return True
        return False
    
    def build_query(self, filters: FilterType) -> str:
        """
        Build a filter expression string from a filter object.
        
        Args:
            filters: Filter object (dict, FilterExpression, or CompoundFilter)
            
        Returns:
            Filter expression string
        """
        if isinstance(filters, dict):
            return self._build_simple_query(filters)
        elif isinstance(filters, FilterExpression):
            return self._build_expression_query(filters)
        elif isinstance(filters, CompoundFilter):
            return self._build_compound_query(filters)
        else:
            return str(filters)
    
    def _build_simple_query(self, filters: Dict[str, Any]) -> str:
        """Build query string from simple dict filters."""
        parts = []
        
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle nested dictionaries (e.g., custom metadata)
                for subkey, subvalue in value.items():
                    parts.append(f'{key}.{subkey} = "{subvalue}"')
            else:
                parts.append(f'{key} = "{value}"')
        
        return ' AND '.join(parts)

    def _build_expression_query(self, expr: FilterExpression) -> str:
        """Build query string from FilterExpression."""
        return f'{expr.field} {expr.operator} "{expr.value}"'
    
    def _build_compound_query(self, compound: CompoundFilter) -> str:
        """Build query string from CompoundFilter."""
        if compound.and_conditions:
            conditions = [self.build_query(cond) for cond in compound.and_conditions]
            return ' AND '.join(conditions)
        elif compound.or_conditions:
            conditions = [self.build_query(cond) for cond in compound.or_conditions]
            return ' OR '.join(conditions)
        else:
            return ""


class FilterParser:
    """Legacy filter parser for backward compatibility."""
    
    def __init__(self):
        """Initialize the filter parser."""
        self.advanced_parser = AdvancedFilterParser()
    
    def parse(self, filter_expr: str) -> Dict[str, Any]:
        """
        Parse a filter expression string into a metadata filter dictionary.
        
        Args:
            filter_expr: Filter expression string (e.g., 'source = "file.pdf"')
            
        Returns:
            Dictionary representing the filter criteria
        """
        result = self.advanced_parser.parse(filter_expr)
        if isinstance(result, dict):
            return result
        else:
            # Convert to dict for backward compatibility
            return self._filter_to_dict(result)
    
    def _filter_to_dict(self, filter_obj: FilterType) -> Dict[str, Any]:
        """Convert filter object to dict for backward compatibility."""
        if isinstance(filter_obj, FilterExpression):
            return {filter_obj.field: filter_obj.value}
        elif isinstance(filter_obj, CompoundFilter):
            # Flatten compound filter to simple dict (lossy conversion)
            result = {}
            if filter_obj.and_conditions:
                for cond in filter_obj.and_conditions:
                    if isinstance(cond, FilterExpression):
                        result[cond.field] = cond.value
            return result
        else:
            return filter_obj
    
    def build_query(self, filters: Dict[str, Any]) -> str:
        """
        Build a filter expression string from a filter dictionary.
        
        Args:
            filters: Filter dictionary
            
        Returns:
            Filter expression string
        """
        return self.advanced_parser.build_query(filters)


# Convenience functions
def parse_filter(filter_input: Union[str, Dict, List]) -> FilterType:
    """
    Parse a filter input (Langbase-style).
    
    Args:
        filter_input: Filter input (string, dict, or list)
        
    Returns:
        Parsed filter object
    """
    parser = AdvancedFilterParser()
    return parser.parse(filter_input)


def build_filter(filters: FilterType) -> str:
    """
    Build a filter expression string from a filter object.
    
    Args:
        filters: Filter object
        
    Returns:
        Filter expression string
    """
    parser = AdvancedFilterParser()
    return parser.build_query(filters)


def parse_simple_filter(filter_expr: str) -> Dict[str, Any]:
    """
    Parse a simple filter expression string (legacy function).
    
    Args:
        filter_expr: Filter expression string
        
    Returns:
        Filter dictionary
    """
    parser = FilterParser()
    return parser.parse(filter_expr)