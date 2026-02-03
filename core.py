"""Core utilities: Git operations, file reading."""
import re
import shutil
from pathlib import Path
from git import Repo, GitCommandError
from config import IGNORE_PATTERNS, MAX_FILE_SIZE, LANG_MAP


class GitManager:
    """Git operations for cloning and managing repositories."""

    def extract_repo_name(self, url: str) -> str:
        """Extract repo name from GitHub URL."""
        patterns = [
            r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$',
            r'gitlab\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$',
            r'bitbucket\.org[:/]([^/]+)/([^/]+?)(?:\.git)?/?$',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return f"{match.group(1)}__{match.group(2)}"
        return url.rstrip('/').split('/')[-1].replace('.git', '')

    def clone_or_pull(self, url: str, repos_dir: str, force: bool = False) -> str:
        """Clone repository or pull if exists."""
        repo_name = self.extract_repo_name(url)
        repo_path = Path(repos_dir) / repo_name

        if force and repo_path.exists():
            print(f"  Removing existing: {repo_path}")
            shutil.rmtree(repo_path)

        if repo_path.exists():
            print(f"  Pulling latest...")
            try:
                repo = Repo(repo_path)
                repo.remotes.origin.pull()
            except GitCommandError as e:
                print(f"  Pull failed, using existing: {e}")
        else:
            print(f"  Cloning {url}...")
            Repo.clone_from(url, repo_path, depth=1)

        return str(repo_path)

    def get_files(self, repo_path: str) -> list:
        """Get all parseable files, respecting ignore patterns."""
        files = []
        repo = Path(repo_path)

        for f in repo.rglob('*'):
            if not f.is_file():
                continue

            rel_path = str(f.relative_to(repo))

            # Check ignore patterns
            if any(re.search(p, rel_path) for p in IGNORE_PATTERNS):
                continue

            # Check file size
            try:
                if f.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            # Check if we support this extension
            if f.suffix.lower() in LANG_MAP:
                files.append(rel_path)

        return sorted(files)


def read_file(path: str) -> str:
    """Read file with encoding fallback."""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']
    for enc in encodings:
        try:
            return Path(path).read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            return ""
    return ""


def get_language(file_path: str) -> str:
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()
    return LANG_MAP.get(ext, 'text')
