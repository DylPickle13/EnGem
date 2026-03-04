from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure repository root is on sys.path so top-level modules (like config)
# can be imported when this module is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from config import SSH_KEY_PATH, REPO_PATH

BRANCH = "master"
REMOTE_NAME = "origin"
# SSH_KEY_PATH is provided by config.py (reads DEPLOY_SSH_KEY env var)


def git_push(message: str) -> str:
	"""
	Commit all changes in the repository with the given message and push to the remote.
	"""
	# Load GitPython while avoiding importing from this package's directory
	current_dir = Path(__file__).resolve().parent
	original_sys_path = list(sys.path)

	try:
		sys.path = [
			entry
			for entry in sys.path
			if Path(entry or ".").resolve() != current_dir
		]
		git_module = importlib.import_module("git")
		git_exc_module = importlib.import_module("git.exc")
	finally:
		sys.path = original_sys_path

	Repo = git_module.Repo
	GitCommandError = git_exc_module.GitCommandError

	repo_path = Path(REPO_PATH).expanduser().resolve()
	if not repo_path.exists():
		raise RuntimeError(f"REPO_PATH does not exist: {repo_path}")

	try:
		repo = Repo(repo_path)
	except getattr(git_exc_module, "InvalidGitRepositoryError", Exception) as e:
		raise RuntimeError(
			f"Invalid git repository at {repo_path}. Ensure REPO_PATH points to a git repository."
		) from e

	if repo.bare:
		raise ValueError("Repository is bare and cannot be committed/pushed.")

	repo.git.add("--all")

	if not repo.is_dirty(untracked_files=True):
		return "No changes to commit."

	repo.index.commit(message)

	ssh_command = None
	if SSH_KEY_PATH:
		expanded_key_path = str(Path(SSH_KEY_PATH).expanduser().resolve())
		ssh_command = (
			f"ssh -i {expanded_key_path} "
			"-o IdentitiesOnly=yes "
			"-o StrictHostKeyChecking=accept-new"
		)

	push_ref = f"HEAD:{BRANCH}"

	try:
		if ssh_command:
			with repo.git.custom_environment(GIT_SSH_COMMAND=ssh_command):
				repo.remote(REMOTE_NAME).push(push_ref)
		else:
			repo.remote(REMOTE_NAME).push(push_ref)
	except GitCommandError as error:
		raise RuntimeError(f"Push failed: {error}") from error

	return f"Committed and pushed to {REMOTE_NAME}/{BRANCH}."


if __name__ == "__main__":
	print(git_push("Resetting"))