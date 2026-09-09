"""Real Windows access checks in disposable directories, never the user repository."""
import sys
import tempfile
import unittest
from pathlib import Path
from code_agent.models import SandboxMode
from code_agent.services.windows_sandbox import WindowsRestrictedTokenSandbox, ShellExecutionSpec, SandboxUnavailableError


@unittest.skipUnless(sys.platform == 'win32', 'Windows API integration')
class WindowsSandboxLiveTests(unittest.TestCase):
    def test_write_and_protected_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'project'
            root.mkdir()
            # Python's private temp directories use OWNER RIGHTS ACEs. Use an
            # explicit user ACE like a normal project checkout for this fixture.
            import win32api, win32con, win32security, ntsecuritycon
            token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
            user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
            descriptor = win32security.GetNamedSecurityInfo(str(root), 1, 4)
            acl = descriptor.GetSecurityDescriptorDacl()
            acl.AddAccessAllowedAceEx(4, 3, ntsecuritycon.FILE_ALL_ACCESS, user)
            win32security.SetNamedSecurityInfo(str(root), 1, 4, None, None, acl, None)
            (root / '.git').mkdir()
            (root / '.git' / 'config').write_text('original')
            (root / '.env').write_text('not-a-real-secret')
            outside = Path(temp) / 'outside.txt'
            outside.write_text('original')
            sandbox = WindowsRestrictedTokenSandbox(root)
            def run(command, mode=SandboxMode.workspace_write):
                return sandbox.run(ShellExecutionSpec(command, root, 10_000, mode))
            result = run("Set-Content ordinary.txt hello; Get-Content ordinary.txt")
            self.assertEqual(result.exit_code, 0, result)
            self.assertTrue((root / 'ordinary.txt').exists(), result)
            self.assertIn('original', run('Get-Content .git/config').stdout)
            result = run('Remove-Item ordinary.txt')
            self.assertEqual(result.exit_code, 0, result)
            self.assertFalse((root / 'ordinary.txt').exists())
            for command in ["Set-Content .git/config changed", "Remove-Item .git -Recurse -Force",
                            "Set-Content .env changed", "Set-Content ../outside.txt changed"]:
                result = run(command)
                self.assertTrue(result.sandbox_denied or result.exit_code != 0, result)
                self.assertEqual((root / '.git' / 'config').read_text(), 'original')
                self.assertEqual(outside.read_text(), 'original')
                self.assertNotIn('not-a-real-secret', result.stdout)
            result = run('Set-Content forbidden.txt hello', SandboxMode.read_only)
            self.assertFalse((root / 'forbidden.txt').exists(), result)
            with self.assertRaises(SandboxUnavailableError):
                run('Write-Output bad', SandboxMode.danger_full_access)
