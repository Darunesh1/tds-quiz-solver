"""Tools module - FIXED naming for agent.py compatibility."""

# ✅ MATCH ACTUAL FUNCTION NAMES from each file
from .add_dependencies import add_dependencies as adddependencies
from .download_file import download_file as downloadfile
from .list_dependencies import list_dependencies as listdependencies
from .run_code import run_code as runcode
from .send_request import postrequest

# ✅ FIX: Import 'get_rendered_html' and alias it to 'getrenderedhtml'
from .web_scraper import get_rendered_html as getrenderedhtml

__all__ = [
    "adddependencies",
    "downloadfile",
    "listdependencies",
    "runcode",
    "postrequest",
    "getrenderedhtml",
]
