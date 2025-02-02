# Open a new PowerShell window and activate the environment
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    conda activate ./env;


  # Change to the parent directory
    cd ..;
    
    # Run the uvicorn server
    uvicorn app.main:app --reload"
