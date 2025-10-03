import sys
from pathlib import Path

def _get_root_dir():
    """Get project root directory reliably across environments"""
    if '__file__' in globals():
        # Start from this file's location
        current_path = Path(__file__).resolve()
    else:
        # Start from current working directory (e.g., in Jupyter)
        current_path = Path.cwd().resolve()
    
    # Go up until we find 'segment_anything' folder
    while current_path != current_path.parent:  # Stop at root
        if (current_path / ".gitignore").exists():
            return current_path
        current_path = current_path.parent
    
    raise FileNotFoundError(
        "Could not find 'segment_anything' folder in any parent directory"
    )

def _setup_project_imports():
    """Setup imports for the project"""
    root_dir = _get_root_dir()
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    return root_dir

# Auto-setup when imported
root_dir = _setup_project_imports()

# Make root_dir the primary export
__all__ = ['root_dir']

def get_root_dir():
    """Public access to root directory function"""
    return _get_root_dir()

"""
USAGE GUIDE:
============

1. Import in any notebook/script:
   from setup_imports import root_dir

2. root_dir is already a Path object - use directly:
   ✅ model_path = root_dir / "model_checkpoints" / "sam_vit_b_01ec64.pth"
   ✅ data_path = root_dir / "Processed_data" / "Healthy"
   ❌ Path(root_dir / "my_path")  # Don't wrap in Path() again

3. Common usage patterns:
   - File paths: checkpoint_path = root_dir / "model_checkpoints" / "sam_vit_b_01ec64.pth"
   - Check existence: checkpoint_path.exists()
   - Convert to string: str(checkpoint_path)  # When APIs need strings
   - List files: list((root_dir / "folder").glob("*.pth"))

4. Example:
   from setup_imports import root_dir
   from segment_anything import sam_model_registry
   
   checkpoint = root_dir / "model_checkpoints" / "sam_vit_b_01ec64.pth"
   model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
"""