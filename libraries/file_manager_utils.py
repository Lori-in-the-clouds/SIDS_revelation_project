import subprocess
from datetime import datetime
import os
from pathlib import Path

def save_as_pdf(notebook):
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"{notebook.parent}/reports"
    output_name = notebook.name.split('.ipynb')[0]
    output_file = f"{output_dir}/{output_name}({today}).pdf"
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "pdf",
        str(notebook),
        "--output", output_file,
    ])

    for file in Path(output_dir).iterdir():
        if file.is_file() and file.suffix != ".pdf":
            file.unlink()