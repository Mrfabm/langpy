"""
Template Engine for Workflow Configuration

Provides Jinja2-style template resolution for dynamic workflow configurations.
Supports filters, conditionals, and context-aware rendering.
"""

import re
import json
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime
import time

try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


class TemplateEngine:
    """Enhanced template engine with Jinja2-style features."""
    
    def __init__(self, use_jinja2: bool = True):
        self.use_jinja2 = use_jinja2 and HAS_JINJA2
        
        # Custom filters
        self.filters = {
            'upper': lambda x: str(x).upper(),
            'lower': lambda x: str(x).lower(),
            'default': lambda x, d: x if x is not None else d,
            'json': lambda x: json.dumps(x),
            'length': lambda x: len(x) if hasattr(x, '__len__') else 0,
            'first': lambda x: x[0] if x and hasattr(x, '__getitem__') else None,
            'last': lambda x: x[-1] if x and hasattr(x, '__getitem__') else None,
            'join': lambda x, sep=',': sep.join(str(i) for i in x) if hasattr(x, '__iter__') else str(x),
            'replace': lambda x, old, new: str(x).replace(old, new),
            'truncate': lambda x, length=50: str(x)[:length] + '...' if len(str(x)) > length else str(x)
        }
        
        # Custom functions
        self.functions = {
            'now': lambda: datetime.now().isoformat(),
            'timestamp': lambda: int(time.time()),
            'uuid': lambda: __import__('uuid').uuid4().hex,
            'random': lambda: __import__('random').random(),
            'env': lambda key, default=None: __import__('os').getenv(key, default)
        }
        
        if self.use_jinja2:
            self.jinja_env = jinja2.Environment(
                loader=jinja2.DictLoader({}),
                undefined=jinja2.StrictUndefined
            )
            
            # Register custom filters
            for name, func in self.filters.items():
                self.jinja_env.filters[name] = func
            
            # Register custom functions
            for name, func in self.functions.items():
                self.jinja_env.globals[name] = func
    
    def render(self, template: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """
        Render a template with the given context.
        
        Args:
            template: Template string or dictionary containing templates
            context: Context variables for template rendering
            
        Returns:
            Rendered template with values substituted
        """
        if isinstance(template, str):
            return self._render_string(template, context)
        elif isinstance(template, dict):
            return self._render_dict(template, context)
        elif isinstance(template, list):
            return self._render_list(template, context)
        else:
            return template
    
    def _render_string(self, template: str, context: Dict[str, Any]) -> str:
        """Render a string template."""
        if not template or not isinstance(template, str):
            return template
        
        # Check if template contains template syntax
        if not self._has_template_syntax(template):
            return template
        
        if self.use_jinja2:
            try:
                jinja_template = self.jinja_env.from_string(template)
                return jinja_template.render(**context)
            except Exception as e:
                # Fall back to simple replacement
                return self._simple_render(template, context)
        else:
            return self._simple_render(template, context)
    
    def _render_dict(self, template: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Render a dictionary template."""
        result = {}
        for key, value in template.items():
            # Render both key and value
            rendered_key = self.render(key, context)
            rendered_value = self.render(value, context)
            result[rendered_key] = rendered_value
        return result
    
    def _render_list(self, template: list, context: Dict[str, Any]) -> list:
        """Render a list template."""
        return [self.render(item, context) for item in template]
    
    def _has_template_syntax(self, template: str) -> bool:
        """Check if string contains template syntax."""
        return ('{{' in template and '}}' in template) or ('{%' in template and '%}' in template)
    
    def _simple_render(self, template: str, context: Dict[str, Any]) -> str:
        """Simple template rendering without Jinja2."""
        result = template
        
        # Simple variable substitution: {{variable}}
        def replace_var(match):
            var_path = match.group(1).strip()
            return str(self._get_nested_value(context, var_path))
        
        result = re.sub(r'\{\{\s*([^}]+)\s*\}\}', replace_var, result)
        
        # Simple filter support: {{variable | filter}}
        def replace_filtered_var(match):
            var_path = match.group(1).strip()
            filter_name = match.group(2).strip()
            
            value = self._get_nested_value(context, var_path)
            
            if filter_name in self.filters:
                return str(self.filters[filter_name](value))
            return str(value)
        
        result = re.sub(r'\{\{\s*([^|]+)\s*\|\s*([^}]+)\s*\}\}', replace_filtered_var, result)
        
        return result
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        if not path:
            return ""
        
        try:
            keys = path.split('.')
            result = data
            
            for key in keys:
                if isinstance(result, dict):
                    result = result.get(key)
                elif hasattr(result, key):
                    result = getattr(result, key)
                else:
                    return ""
                
                if result is None:
                    return ""
            
            return result
        except (KeyError, AttributeError, TypeError):
            return ""
    
    def add_filter(self, name: str, func: Callable) -> None:
        """Add a custom filter."""
        self.filters[name] = func
        
        if self.use_jinja2:
            self.jinja_env.filters[name] = func
    
    def add_function(self, name: str, func: Callable) -> None:
        """Add a custom function."""
        self.functions[name] = func
        
        if self.use_jinja2:
            self.jinja_env.globals[name] = func


# Global template engine instance
_template_engine: Optional[TemplateEngine] = None


def get_template_engine() -> TemplateEngine:
    """Get the global template engine instance."""
    global _template_engine
    if _template_engine is None:
        _template_engine = TemplateEngine()
    return _template_engine


def render_template(template: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
    """Convenience function to render templates."""
    engine = get_template_engine()
    return engine.render(template, context)


def render_step_config(config: Optional[Dict[str, Any]], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Render step configuration templates."""
    if not config:
        return config
    
    engine = get_template_engine()
    return engine.render(config, context)


# Example usage and testing
if __name__ == "__main__":
    # Test the template engine
    engine = TemplateEngine()
    
    context = {
        'user': {'name': 'John', 'type': 'premium'},
        'step1': {'output': 'Hello World', 'status': 'completed'},
        'items': ['apple', 'banana', 'cherry'],
        'score': 85
    }
    
    # Test simple variable substitution
    template1 = "Hello {{user.name}}, your score is {{score}}"
    result1 = engine.render(template1, context)
    print(f"Template 1: {result1}")
    
    # Test filters
    template2 = "{{step1.output | upper}} - {{items | join:' and '}}"
    result2 = engine.render(template2, context)
    print(f"Template 2: {result2}")
    
    # Test functions
    template3 = "Generated at {{now()}} with ID {{uuid()}}"
    result3 = engine.render(template3, context)
    print(f"Template 3: {result3}")
    
    # Test dictionary rendering
    template_dict = {
        "input": "Process: {{step1.output | lower}}",
        "user_info": {
            "name": "{{user.name}}",
            "type": "{{user.type | default:'standard'}}"
        },
        "metadata": {
            "timestamp": "{{now()}}",
            "item_count": "{{items | length}}"
        }
    }
    result_dict = engine.render(template_dict, context)
    print(f"Template Dict: {json.dumps(result_dict, indent=2)}") 