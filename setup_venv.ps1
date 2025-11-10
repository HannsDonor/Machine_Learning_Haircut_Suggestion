# setup_venv.ps1
# PowerShell script to create a venv and install all packages

$venvName = "venv"

# Remove existing venv if exists
if (Test-Path $venvName) {
    Write-Host "Removing existing venv..."
    Remove-Item -Recurse -Force $venvName
}

# Create new venv
Write-Host "Creating virtual environment..."
python -m venv $venvName

# Activate venv
Write-Host "Activating virtual environment..."
& .\$venvName\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install packages from requirements.txt
if (Test-Path "requirements.txt") {
    Write-Host "Installing packages from requirements.txt..."
    pip install --no-cache-dir -r requirements.txt
} else {
    Write-Host "❌ requirements.txt not found in current folder!"
}

Write-Host "✅ Setup complete. Virtual environment ready."
Write-Host "Activate with: .\$venvName\Scripts\Activate.ps1"
