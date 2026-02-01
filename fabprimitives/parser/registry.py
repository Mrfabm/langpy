from typing import Dict, Type
from .sync_parser import SyncParser
from .async_parser import AsyncParser

# Registry for filetype to parser class
PARSER_REGISTRY: Dict[str, Type] = {}

def register_parser(filetype: str, parser_cls: Type):
    PARSER_REGISTRY[filetype] = parser_cls

def get_parser_for_filetype(filetype: str):
    return PARSER_REGISTRY.get(filetype)

# Register default parsers
register_parser('pdf', SyncParser)
register_parser('docx', SyncParser)
register_parser('html', SyncParser)
register_parser('txt', SyncParser) 