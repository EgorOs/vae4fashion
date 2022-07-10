# Setup

### Windows installation note

To train on Windows machine with GPU run this line after poetry install.
```bash
pip install -r requirements.txt
pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

After that training with GPU will likely fail, [this answer](https://stackoverflow.com/a/63236882) resolved the problem.