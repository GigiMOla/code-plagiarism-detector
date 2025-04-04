import os
import subprocess
import httpx
from tqdm import tqdm
import time
from config.settings import *

def clone_repositories():
    """Your existing clone function"""
    print("Checking repositories...")
    with open("/config/repos.txt") as f:
        repos = [r.strip() for r in f.readlines() if r.strip()]

    os.makedirs("/repositories", exist_ok=True)

    for repo in repos:
        if not repo or not repo.startswith(('http', 'git@')):
            continue

        repo_name = repo.split('/')[-1].replace('.git', '')
        target_dir = os.path.join("/repositories", repo_name)

        if os.path.exists(target_dir):
            print(f"Repository {repo_name} already exists, skipping...")
            continue

        print(f"Cloning {repo_name}...")
        try:
            subprocess.run(
                ['git', 'clone', '--depth', '1', repo, target_dir],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo}: {e.stderr.decode().strip()}")
            continue

def get_code_files():
    """Your existing file discovery function"""
    all_extensions = []
    for lang_config in LANGUAGE_CONFIG.values():
        all_extensions.extend(lang_config['extensions'])

    code_files = []
    for root, _, files in os.walk("/repositories"):
        for file in files:
            if any(file.endswith(ext) for ext in all_extensions):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if len(content.split('\n')) >= PROCESSING_CONFIG['min_code_lines']:
                        code_files.append(full_path)
                except (UnicodeDecodeError, PermissionError, IOError) as e:
                    print(f"Skipping {full_path}: {str(e)}")
                    continue
    return code_files

async def process_repositories():
    """Process all repositories and index them"""
    await wait_for_service("http://embedding-service:8001/health", timeout=300)
    await wait_for_service("http://vector-db:8002/health", timeout=60)

    files = get_code_files()
    if not files:
        print("No code files found!")
        return

    print(f"Processing {len(files)} files...")

    async with httpx.AsyncClient() as client:
        for file in tqdm(files, desc="Indexing files"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Determine language
                language = 'python'
                for lang, config in LANGUAGE_CONFIG.items():
                    if any(file.endswith(ext) for ext in config['extensions']):
                        language = lang
                        break

                # Get embedding
                embed_resp = await client.post(
                    "http://embedding-service:8001/embed",
                    json={"text": content, "language": language},
                    timeout=30
                )
                embedding = embed_resp.json()["embedding"]

                # Store in vector DB
                await client.post(
                    "http://vector-db:8002/add",
                    json={
                        "embedding": embedding,
                        "metadata": {
                            "path": file,
                            "language": language,
                            "original_length": len(content.split('\n'))
                        }
                    },
                    timeout=30
                )

            except Exception as e:
                print(f"Failed to process {file}: {str(e)}")
                continue

async def wait_for_service(url: str, timeout: int = 60):
    """Wait for a service to become available"""
    start = time.time()
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass

            if time.time() - start > timeout:
                raise TimeoutError(f"Service at {url} not ready after {timeout}s")
            
            time.sleep(1)

if __name__ == "__main__":
    import asyncio
    clone_repositories()
    asyncio.run(process_repositories())