$ErrorActionPreference = "Stop"
Write-Host "Installing CORTEX final dependencies..."
python -m pip install -r requirements.final.txt
Write-Host "Running project preflight..."
python -m backend.scripts.preflight_final
Write-Host "Creating final application tables..."
python -m backend.scripts.upgrade_final_schema
Write-Host "Updating demo users and Agent Passports..."
python -m backend.init_db
Write-Host "Preparing persistent LangGraph checkpoints..."
python -m backend.scripts.setup_checkpointer
Write-Host "Running syntax validation..."
python -m compileall backend frontend
Write-Host "CORTEX final setup completed."
