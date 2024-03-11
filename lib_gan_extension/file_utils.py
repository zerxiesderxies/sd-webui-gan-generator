import os
from modules import shared
import platform
import subprocess as sp

def mkdir_p(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # If the directory already exists, it's okay
        pass
    except OSError as e:
        # Handle other errors
        print(f"Error creating directory: {e}")

def open_folder(f, images=None, index=None):
    if shared.cmd_opts.hide_ui_dir_config:
        return

    try:
        if 'Sub' in shared.opts.open_dir_button_choice:
            image_dir = os.path.split(images[index]["name"].rsplit('?', 1)[0])[0]
            if 'temp' in shared.opts.open_dir_button_choice or not ui_tempdir.is_gradio_temp_path(image_dir):
                f = image_dir
    except Exception:
        pass

    if not os.path.exists(f):
        msg = f'Folder "{f}" does not exist. After you create an image, the folder will be created.'
        print(msg)
        gr.Info(msg)
        return
    elif not os.path.isdir(f):
        msg = f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
"""
        print(msg, file=sys.stderr)
        gr.Warning(msg)
        return

    path = os.path.normpath(f)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        sp.Popen(["open", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        sp.Popen(["wsl-open", path])
    else:
        sp.Popen(["xdg-open", path])
