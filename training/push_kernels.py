from pathlib import Path
import os
import subprocess
import shutil

ROOT = Path(__file__).resolve().parent

def main():
    token_path = Path("/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token")
    if not token_path.exists():
        print("HF token not found at", token_path)
        return
        
    with token_path.open("r") as f:
        token = f.read().strip()
        
    # Scripts to push
    scripts = [
        ("kaggle_train_gnn.py", "gnn-kernel-metadata.json"),
        ("kaggle_train_edl.py", "edl-kernel-metadata.json")
    ]
    
    # We must push them sequentially
    for script_name, metadata_name in scripts:
        print(f"Preparing to push {script_name} using metadata {metadata_name}...")
        
        # 1. Back up original script
        script_path = ROOT / script_name
        metadata_path = ROOT / metadata_name
        backup_name = script_path.with_suffix(script_path.suffix + ".bak")
        shutil.copy(script_path, backup_name)
        
        # 2. Back up original kernel-metadata.json
        meta_backup = None
        kernel_metadata = ROOT / "kernel-metadata.json"
        if kernel_metadata.exists():
            meta_backup = ROOT / "kernel-metadata.json.bak"
            shutil.copy(kernel_metadata, meta_backup)
            
        try:
            # 3. Read and replace placeholder in script
            with script_path.open("r") as f:
                content = f.read()
                
            modified_content = content.replace("PLACEHOLDER_HF_TOKEN", token)
            
            with script_path.open("w") as f:
                f.write(modified_content)
                
            # 4. Copy metadata to kernel-metadata.json
            shutil.copy(metadata_path, kernel_metadata)
                
            # 5. Push to Kaggle
            print(f"Pushing {script_name} to Kaggle...")
            res = subprocess.run(["kaggle", "kernels", "push", "-p", str(ROOT)], capture_output=True, text=True)
            print("Stdout:", res.stdout)
            print("Stderr:", res.stderr)
            
        finally:
            # 6. Restore original script
            if os.path.exists(backup_name):
                shutil.move(backup_name, script_path)
                
            # 7. Restore original kernel-metadata.json
            if meta_backup and meta_backup.exists():
                shutil.move(meta_backup, kernel_metadata)
            elif kernel_metadata.exists():
                os.remove(kernel_metadata)
                
        print(f"Finished pushing {script_name}.\n")

if __name__ == "__main__":
    main()
