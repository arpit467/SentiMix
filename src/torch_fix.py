import os
import sys
import ctypes

def init_torch_dlls():
    """Preloads necessary Windows DLLs for PyTorch."""
    if sys.platform.startswith("win"):
        # Look for torch lib directory
        candidates = [
            os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib"),
            r"C:\Users\ARPIT\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
        ]
        for torch_lib in candidates:
            if os.path.exists(torch_lib):
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(torch_lib)
                    except Exception:
                        pass
                for dll_name in ["c10.dll", "torch_cpu.dll"]:
                    dll_path = os.path.join(torch_lib, dll_name)
                    if os.path.exists(dll_path):
                        try:
                            ctypes.CDLL(dll_path)
                        except Exception:
                            pass
                break

init_torch_dlls()
