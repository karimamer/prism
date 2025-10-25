try:
    from .unified_system import UnifiedEntityResolutionSystem
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings

    warnings.warn(f"Could not import UnifiedEntityResolutionSystem: {e}", stacklevel=2)
    UnifiedEntityResolutionSystem = None
