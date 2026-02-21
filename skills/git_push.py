from __future__ import annotations

import importlib
import sys
from pathlib import Path
from config import SSH_KEY_PATH, REPO_PATH

BRANCH = "master"
REMOTE_NAME = "origin"
# SSH_KEY_PATH is provided by config.py (reads DEPLOY_SSH_KEY env var)


def _load_gitpython():
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

	return git_module.Repo, git_exc_module.GitCommandError


def git_push(message: str) -> str:
	"""
	Commit all changes in the repository with the given message and push to the remote.
	"""
	Repo, GitCommandError = _load_gitpython()
	repo = Repo(REPO_PATH)

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
