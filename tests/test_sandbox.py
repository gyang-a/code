from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from code_agent.services.sandbox import DockerSandbox, SandboxPolicy, ShellSandbox, build_shell_sandbox
from code_agent.services.workspace import Workspace, WorkspaceError


class SandboxTests(unittest.TestCase):
    def test_rejects_obvious_path_escape_in_argv(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            sandbox = ShellSandbox(workspace)

            with self.assertRaises(WorkspaceError):
                sandbox.run(["python", "../outside.py"], timeout=5)

    def test_allows_workspace_relative_script_path(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            sandbox = ShellSandbox(workspace)

            result = sandbox.run(["python", "-c", "print('ok')"], timeout=5)

            self.assertEqual(result.returncode, 0)
            self.assertIn("ok", result.stdout)

    def test_build_shell_sandbox_selects_docker_backend(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            sandbox = build_shell_sandbox(
                workspace,
                SandboxPolicy(backend="docker", docker_image="python:3.12-slim"),
            )

            self.assertIsInstance(sandbox, DockerSandbox)

    def test_docker_backend_wraps_command_in_disposable_container(self) -> None:
        with TemporaryWorkspace() as tmp_path:
            workspace = Workspace(tmp_path)
            sandbox = DockerSandbox(
                workspace,
                SandboxPolicy(backend="docker", docker_image="python:3.12-slim"),
            )

            with patch("code_agent.services.sandbox.shutil.which", return_value="docker"), patch(
                "code_agent.services.sandbox.subprocess.run"
            ) as run_mock:
                run_mock.return_value.returncode = 0
                run_mock.return_value.stdout = "ok\n"
                run_mock.return_value.stderr = ""

                result = sandbox.run(["python", "-c", "print('ok')"], timeout=5)

            docker_args = run_mock.call_args.args[0]
            self.assertEqual(result.returncode, 0)
            self.assertEqual(docker_args[:3], ["docker", "run", "--rm"])
            self.assertIn("--network", docker_args)
            self.assertIn("none", docker_args)
            self.assertIn("python:3.12-slim", docker_args)
            self.assertEqual(docker_args[-3:], ["python", "-c", "print('ok')"])
            self.assertIn(f"{tmp_path.resolve()}:/workspace", docker_args)


class TemporaryWorkspace:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._tmp.cleanup()
