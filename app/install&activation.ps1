# Create the conda environment with Python 3.9
conda create --prefix ./env python=3.9 -y

# Open a new PowerShell window and activate the environment
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    conda activate ./env;
    pip install -r requirements.txt;
    Write-Host 'Environment setup complete!';
    
    # Change to the parent directory
    cd ..;
    
    # Run the uvicorn server
    uvicorn app.main:app --reload"
