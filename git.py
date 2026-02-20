from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


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


def commit_and_push(
	repo_path: str,
	message: str,
	branch: str = "main",
	remote_name: str = "origin",
	ssh_key_path: str | None = None,
) -> str:
	Repo, GitCommandError = _load_gitpython()
	repo = Repo(repo_path)

	if repo.bare:
		raise ValueError("Repository is bare and cannot be committed/pushed.")

	repo.git.add("--all")

	if not repo.is_dirty(untracked_files=True):
		return "No changes to commit."

	repo.index.commit(message)

	ssh_command = None
	if ssh_key_path:
		expanded_key_path = str(Path(ssh_key_path).expanduser().resolve())
		ssh_command = (
			f"ssh -i {expanded_key_path} "
			"-o IdentitiesOnly=yes "
			"-o StrictHostKeyChecking=accept-new"
		)

	push_ref = f"HEAD:{branch}"

	try:
		if ssh_command:
			with repo.git.custom_environment(GIT_SSH_COMMAND=ssh_command):
				repo.remote(remote_name).push(push_ref)
		else:
			repo.remote(remote_name).push(push_ref)
	except GitCommandError as error:
		raise RuntimeError(f"Push failed: {error}") from error

	return f"Committed and pushed to {remote_name}/{branch}."


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Commit and push changes using GitPython and an optional SSH key."
	)
	parser.add_argument(
		"--repo-path",
		default=".",
		help="Path to the local git repository (default: current directory).",
	)
	parser.add_argument(
		"--message",
		required=True,
		help="Commit message.",
	)
	parser.add_argument(
		"--branch",
		default="main",
		help="Branch to push to (default: main).",
	)
	parser.add_argument(
		"--remote",
		default="origin",
		help="Remote name (default: origin).",
	)
	parser.add_argument(
		"--ssh-key",
		default=os.getenv("DEPLOY_SSH_KEY", "~/.ssh/id_ed25519"),
		help="Path to SSH private key (default: DEPLOY_SSH_KEY or ~/.ssh/id_ed25519).",
	)
	return parser


if __name__ == "__main__":
	parser = _build_parser()
	args = parser.parse_args()

	result = commit_and_push(
		repo_path=args.repo_path,
		message=args.message,
		branch=args.branch,
		remote_name=args.remote,
		ssh_key_path=args.ssh_key,
	)
	print(result)
