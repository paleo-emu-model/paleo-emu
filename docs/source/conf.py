# docs/conf.py

import os
import sys
from sphinx_gallery.sorting import FileNameSortKey  # Correct import

# Project information
project = 'paleo-emu'

# Theme
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "navbar_center": ["navbar-nav"],
    "show_toc_level": 2,
    "show_nav_level": 2,
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "navigation_with_keys": False,
    "logo": {
        "text": "🌍 paleo-emu",
    },
    "secondary_sidebar_items": ["page-toc", "sg_download_links", "sg_launcher_links"],
}

# Extensions
extensions = [
    'sphinx_gallery.gen_gallery',    
]

# Sphinx Gallery config
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': r'plot_.*\.py',
    'within_subsection_order': FileNameSortKey,
    'thumbnail_size': (400, 280),
    'download_all_examples': False,  # Disable "Download all examples" button
    'example_extensions': ['.py'],   # Only generate .py files (no downloads)
    'remove_config_comments': True,  # Remove metadata comments
    'image_scrapers': ('matplotlib',),  # Important for plot capture
    'reference_url': {'sklearn': None},  # Cleaner cross-references


    'show_signature': False,  # Don't show function signatures
    'line_numbers': False,  # Don't show line numbers
}

