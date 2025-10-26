"""
Prism - Entity Resolution System.
"""

__version__ = "0.1.0"

# Re-export main components for convenience
from entity_resolution import SystemConfig, UnifiedEntityResolutionSystem

__all__ = [
    "__version__",
    "UnifiedEntityResolutionSystem",
    "SystemConfig",
]
