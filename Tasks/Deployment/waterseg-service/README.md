## 1) go to the service directory
cd "C:\Cellula\Water Segmentation\Project\Water-Segmentation\Tasks\Deployment\waterseg-service"

# 2) (optional) turn off conda base so it doesn’t interfere

conda deactivate

# 3) create a venv (choose one name)
python -m venv2 .venv   

# 4) activate the venv

.\.venv2\Scripts\Activate.ps1

# 5) Install dependencies

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# 6) Run (development server)

python -m flask --app wsgi:app run --debug

