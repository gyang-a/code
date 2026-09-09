"""Conservative PowerShell preflight. Approval never changes the sandbox mode.

The parser only inspects syntax; it never invokes the submitted script. External
programs and dynamic scripts require approval because their effects are unknown.
"""
import json
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from code_agent.models import PermissionDecision, RiskLevel
from code_agent.services.path_policy import is_protected, is_secret

PARSER = r'''
$source = [Console]::In.ReadToEnd()
$tokens = $null; $errors = $null
$tree = [System.Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$errors)
$commands = @($tree.FindAll({param($n) $n -is [System.Management.Automation.Language.CommandAst]}, $true) | ForEach-Object {
    @{name=$_.GetCommandName(); elements=@($_.CommandElements | ForEach-Object {
        @{text=$_.Extent.Text; kind=$_.GetType().Name; value=$(if ($_ -is [System.Management.Automation.Language.StringConstantExpressionAst]) {$_.Value} else {$_.Extent.Text})}
    })}
})
@{errors=@($errors | ForEach-Object {$_.Message}); commands=$commands;
  strings=@($tree.FindAll({param($n) $n -is [System.Management.Automation.Language.StringConstantExpressionAst]}, $true) | ForEach-Object {$_.Value});
  complex=@($tree.FindAll({param($n) $n -is [System.Management.Automation.Language.VariableExpressionAst] -or $n -is [System.Management.Automation.Language.RedirectionAst] -or $n -is [System.Management.Automation.Language.InvokeMemberExpressionAst] -or $n -is [System.Management.Automation.Language.ScriptBlockExpressionAst] -or $n -is [System.Management.Automation.Language.ExpandableStringExpressionAst]}, $true)).Count -gt 0
} | ConvertTo-Json -Depth 8 -Compress
'''

READ_COMMANDS = frozenset({'get-content', 'get-childitem', 'get-item', 'get-location',
    'test-path', 'resolve-path', 'select-string', 'write-output'})
DELETE_COMMANDS = frozenset({'remove-item', 'rm', 'del', 'erase', 'rd', 'rmdir', 'ri'})
BLOCK_COMMANDS = frozenset({'invoke-expression', 'iex', 'set-acl', 'icacls', 'takeown',
    'format-volume', 'format', 'diskpart', 'clear-disk', 'set-executionpolicy'})


def decision(action: str, reason: str) -> PermissionDecision:
    return PermissionDecision(risk={'allow': RiskLevel.level_0, 'ask': RiskLevel.level_2,
        'deny': RiskLevel.level_3}[action], allowed=action == 'allow',
        requires_approval=action == 'ask', reason=reason)


@lru_cache(maxsize=256)
def parse(command: str) -> dict:
    executable = shutil.which('pwsh') or shutil.which('powershell')
    if not executable:
        raise ValueError('PowerShell parser is unavailable')
    result = subprocess.run([executable, '-NoLogo', '-NoProfile', '-NonInteractive',
        '-Command', PARSER], input=command, capture_output=True, text=True,
        encoding='utf-8', timeout=10)
    if result.returncode:
        raise ValueError('PowerShell parser failed')
    return json.loads(result.stdout)


def classify_shell(workspace, args: dict) -> PermissionDecision:
    from code_agent.services.workspace import WorkspaceError
    if args.get('sandbox_permissions') is not None:
        return decision('deny', 'Shell permissions are host-controlled; escalation is disabled.')
    command = args.get('command')
    if not isinstance(command, str) or not command.strip():
        return decision('deny', 'Missing Shell command.')
    try:
        cwd = workspace.resolve(args.get('workdir', '.'))
        if is_secret(cwd) or is_protected(cwd, workspace.root):
            return decision('deny', 'Shell workdir is protected or sensitive.')
        ast = parse(command)
    except (ValueError, TypeError, OSError, subprocess.SubprocessError) as exc:
        return decision('deny', f'Cannot validate Shell command: {exc}')
    if ast.get('errors'):
        return decision('deny', 'PowerShell syntax error; correct the command before execution.')

    # Inspect literal paths even in commands which will require approval.
    for value in ast.get('strings', []):
        if not isinstance(value, str):
            continue
        candidate = value.strip('"\'')
        if candidate.startswith('-') and '=' in candidate:
            candidate = candidate.split('=', 1)[1]
        if re.match(r'^[a-z]+://', candidate, re.I):
            continue
        if candidate.startswith('-'):
            continue
        if ':' in candidate and not re.match(r'^[a-zA-Z]:[\\/]', candidate):
            return decision('deny', f'Provider paths, drive-relative paths and alternate streams are forbidden: {candidate}')
        looks_path = ('/' in candidate or '\\' in candidate or candidate.startswith('.')
                      or is_secret(Path(candidate)))
        if not looks_path or '\n' in candidate:
            continue
        try:
            path = workspace.resolve(cwd / candidate)
        except (WorkspaceError, OSError, ValueError):
            return decision('deny', f'Path outside workspace: {candidate}')
        if is_secret(path) or is_protected(path, workspace.root):
            return decision('deny', f'Protected or sensitive path: {candidate}')

    asks = bool(ast.get('complex')) or any(
        any(char in value for char in '*?[]') for value in ast.get('strings', []))
    destructive = False
    for entry in ast.get('commands', []):
        raw = (entry.get('name') or '').lower()
        name = raw.removesuffix('.exe').removesuffix('.cmd')
        elements = entry.get('elements', [])[1:]
        values = [str(e.get('value', '')) for e in elements]
        if name in BLOCK_COMMANDS or any(v.lower().startswith(('-encodedcommand', '-enc')) for v in values):
            return decision('deny', 'Dynamic execution or security-changing command is forbidden.')
        if name in DELETE_COMMANDS:
            destructive = True
            targets = [v for v in values if not v.startswith('-')]
            if not targets or any(v in {'.', './', '.\\', '*', './*', '.\\*'} for v in targets):
                return decision('deny', 'Deleting the project/current directory or an ambiguous tree is forbidden.')
            for target in targets:
                if workspace.resolve(cwd / target) == workspace.root:
                    return decision('deny', 'Deleting the workspace root is forbidden.')
            asks = True
        elif name not in READ_COMMANDS:
            asks = True
    if not ast.get('commands'):
        asks = True
    policy = workspace.shell_approval_policy
    if (policy == 'on-risk' and not destructive and not ast.get('complex')
            and command in workspace.shell_allowed_commands):
        asks = False
    if policy == 'untrusted' and ast.get('complex'):
        asks = True
    if asks and policy == 'never':
        return decision('deny', 'This command needs approval, but approval policy is never.')
    if asks:
        return decision('ask', 'This command can modify files or execute project code. Approval keeps the current sandbox boundary.')
    return decision('allow', 'Recognized read operation; execute within the host sandbox.')
