python -m pip install --upgrade pip
python -m pip install --ignore-installed "blinker>=1.9.0"
pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_docker.txt
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
cd Depth-Anything-3
pip install -e .
