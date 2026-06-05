"""Compatibility layer for qualibrate import path changes between versions.

New qualibrate versions moved many modules under qualibrate.core.
This module tries old import paths first, then falls back to new paths.
"""

# qualibrate.parameters → qualibrate.core.parameters
try:
    from qualibrate.parameters import GraphParameters, RunnableParameters
except ImportError:
    from qualibrate.core.parameters import GraphParameters, RunnableParameters

# qualibrate.config → qualibrate.core.config
try:
    from qualibrate.config.resolvers import get_quam_state_path
except ImportError:
    from qualibrate.core.config.resolvers import get_quam_state_path

# qualibrate_app → qualibrate.app
try:
    from qualibrate_app.config import get_config_path, get_settings
except ImportError:
    from qualibrate.app.config import get_config_path, get_settings

# qualibrate.utils → qualibrate.core.utils
try:
    from qualibrate.utils.node.path_solver import get_node_dir_path
except ImportError:
    from qualibrate.core.utils.node.path_solver import get_node_dir_path

# qualibrate.basic_orchestrator → qualibrate.core.orchestration.basic_orchestrator
try:
    from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
except ImportError:
    from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator

# qualibrate.qualibration_graph → qualibrate.core.qualibration_graph
try:
    from qualibrate.qualibration_graph import QualibrationGraph
except ImportError:
    from qualibrate.core.qualibration_graph import QualibrationGraph

# qualibrate.qualibration_library → qualibrate.core.qualibration_library
try:
    from qualibrate.qualibration_library import QualibrationLibrary
except ImportError:
    from qualibrate.core.qualibration_library import QualibrationLibrary

try:
    from qualibrate.storage.local_storage_manager import LocalStorageManager
except ImportError:
    from qualibrate.core.storage.local_storage_manager import LocalStorageManager

__all__ = [
    "BasicOrchestrator",
    "GraphParameters",
    "QualibrationGraph",
    "QualibrationLibrary",
    "RunnableParameters",
    "get_config_path",
    "get_node_dir_path",
    "get_quam_state_path",
    "get_settings",
    "LocalStorageManager",
]
