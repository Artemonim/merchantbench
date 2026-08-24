# =============================================================================
# build.ps1 — AE2 orchestrator
# =============================================================================
# Compatible with Windows PowerShell 5.1 and PowerShell 7+.
# Pipeline: self-check -> fmt -> lint -> compile -> test -> coverage ->
#           security -> codebase-memory -> launch -> archive
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Fast,
    [switch]$SkipLaunch,
    [switch]$NoCache,
    [switch]$ForceAll,
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# =============================================================================
# * Configuration (user-tunable)
# =============================================================================
# * Included in every cache key so schema/toolchain changes invalidate stamps.
$script:CacheSchemaVersion = '1'
$script:ReportSchemaVersion = 1
# * Adapter stages that may be hash-cached. test/coverage are never cached.
$script:CacheableStages = @('self-check', 'fmt', 'lint', 'compile')
$script:MutatingStages = @('fmt', 'lint')
# * These failures abort remaining work (test/coverage still record remaining skips).
$script:FailFastStages = @('self-check', 'fmt', 'lint', 'compile')

# =============================================================================
# Bootstrap
# =============================================================================
$script:RepoRoot = $PSScriptRoot
Set-Location -LiteralPath $script:RepoRoot

try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
    $OutputEncoding = New-Object System.Text.UTF8Encoding $false
}
catch {
    # ! Some hosts reject console encoding changes; file writes still use UTF-8.
    Write-Verbose 'Console UTF-8 encoding was not applied.'
}

$script:CacheDir = Join-Path $script:RepoRoot '.ci_cache'
$script:LogsDir = Join-Path $script:CacheDir 'logs'
$script:ReportPath = Join-Path $script:CacheDir 'report.json'
$script:EnforcerDir = Join-Path $script:RepoRoot '.enforcer'
$script:EnforcerLastCheckPath = Join-Path $script:EnforcerDir 'Enforcer_last_check.log'
$script:EnforcerStatsPath = Join-Path $script:EnforcerDir 'Enforcer_stats.log'
$script:BuildPyPath = Join-Path $script:RepoRoot 'build.py'
$script:PssaSettingsPath = Join-Path $script:RepoRoot 'PSScriptAnalyzerSettings.psd1'

$script:NoCache = [bool]($NoCache -or $ForceAll)
$script:Fast = [bool]$Fast
$script:SkipLaunch = [bool]$SkipLaunch
$script:CleanRequested = [bool]$Clean
$script:StartedAt = Get-Date
$script:PipelineException = $null

$script:StageResults = New-Object System.Collections.ArrayList
$script:Issues = New-Object System.Collections.ArrayList
$script:Metrics = [ordered]@{}

function Initialize-CiDirectories {
    if ($script:CleanRequested -and (Test-Path -LiteralPath $script:CacheDir)) {
        Write-Host 'Cleaning .ci_cache/' -ForegroundColor Yellow
        Remove-Item -LiteralPath $script:CacheDir -Recurse -Force
    }
    foreach ($dir in @($script:CacheDir, $script:LogsDir, $script:EnforcerDir)) {
        if (-not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Path $dir | Out-Null
        }
    }
}

function Resolve-ProjectPython {
    $candidates = @(
        (Join-Path $script:RepoRoot '.venv\Scripts\python.exe')
        (Join-Path $script:RepoRoot 'venv\Scripts\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $cmd -and $cmd.Source) {
        return [string]$cmd.Source
    }
    throw 'Python interpreter not found. Create a venv at .venv\Scripts\python.exe (Python 3.11) or ensure python is on PATH.'
}

function Get-ToolVersionString {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [string]$Fallback = 'unavailable'
    )
    try {
        $output = & $FilePath @ArgumentList 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace([string]$output)) {
            return $Fallback
        }
        return ([string]$output).Trim()
    }
    catch {
        return $Fallback
    }
}

function ConvertTo-ProcessArgumentString {
    param([string[]]$ArgumentList)
    $parts = New-Object 'System.Collections.Generic.List[string]'
    foreach ($arg in @($ArgumentList)) {
        if ($null -eq $arg) { continue }
        $text = [string]$arg
        if ($text -notmatch '\s' -and $text -notmatch '"') {
            [void]$parts.Add($text)
        }
        else {
            [void]$parts.Add(('"{0}"' -f ($text -replace '"', '\"')))
        }
    }
    return [string]::Join(' ', $parts)
}

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = $script:RepoRoot
    )
    $outFile = Join-Path $env:TEMP ('mb-ci-out-{0}.txt' -f [guid]::NewGuid().ToString('N'))
    $errFile = Join-Path $env:TEMP ('mb-ci-err-{0}.txt' -f [guid]::NewGuid().ToString('N'))
    $argString = ConvertTo-ProcessArgumentString -ArgumentList $ArgumentList
    $started = Get-Date
    try {
        $proc = Start-Process -FilePath $FilePath -ArgumentList $argString `
            -WorkingDirectory $WorkingDirectory -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $stdoutLines = @()
        $stderrLines = @()
        if (Test-Path -LiteralPath $outFile) {
            $stdoutLines = @(Get-Content -LiteralPath $outFile -Encoding UTF8 -ErrorAction SilentlyContinue)
        }
        if (Test-Path -LiteralPath $errFile) {
            $stderrLines = @(Get-Content -LiteralPath $errFile -Encoding UTF8 -ErrorAction SilentlyContinue)
        }
        $combined = @()
        if ($stdoutLines.Count -gt 0) { $combined += $stdoutLines }
        if ($stderrLines.Count -gt 0) { $combined += $stderrLines }
        return [ordered]@{
            ExitCode   = [int]$proc.ExitCode
            StdOut     = $stdoutLines
            StdErr     = $stderrLines
            Combined   = $combined
            DurationMs = [int]((Get-Date) - $started).TotalMilliseconds
        }
    }
    finally {
        Remove-Item -LiteralPath $outFile, $errFile -Force -ErrorAction SilentlyContinue
    }
}

function Write-StageLog {
    param(
        [Parameter(Mandatory)][string]$StageName,
        [string[]]$Lines = @()
    )
    $logPath = Join-Path $script:LogsDir ('{0}.log' -f $StageName)
    Set-Content -LiteralPath $logPath -Value @($Lines) -Encoding utf8
    return $logPath
}

function ConvertTo-Ae2StageResult {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Status,
        [string]$Note = '',
        [int]$DurationMs = 0,
        [object]$Details = $null
    )
    if ($null -eq $Details) {
        $Details = [ordered]@{}
    }
    return [ordered]@{
        name        = $Name
        status      = $Status
        note        = $Note
        duration_ms = [int]$DurationMs
        details     = $Details
    }
}

function ConvertTo-Ae2Issue {
    param(
        [string]$Language = 'python',
        [string]$Tool = 'ci',
        [string]$Rule = 'unknown',
        [int]$Count = 1,
        [string]$Message = ''
    )
    if ($Count -lt 1) { $Count = 1 }
    return [ordered]@{
        language = $Language
        tool     = $Tool
        rule     = $Rule
        count    = [int]$Count
        message  = $Message
    }
}

function ConvertTo-OrderedHashtable {
    param($InputObject)
    $result = [ordered]@{}
    if ($null -eq $InputObject) { return $result }
    if ($InputObject -is [System.Collections.IDictionary]) {
        foreach ($key in $InputObject.Keys) {
            $result[[string]$key] = $InputObject[$key]
        }
        return $result
    }
    foreach ($prop in $InputObject.PSObject.Properties) {
        $result[$prop.Name] = $prop.Value
    }
    return $result
}

function ConvertTo-IssueFromObject {
    param($Raw)
    if ($null -eq $Raw) { return $null }
    $map = ConvertTo-OrderedHashtable $Raw
    $count = 1
    if ($map.Contains('count') -and $null -ne $map['count']) {
        $count = [int]$map['count']
    }
    $language = 'python'
    if ($map.Contains('language')) { $language = [string]$map['language'] }
    $tool = 'ci'
    if ($map.Contains('tool')) { $tool = [string]$map['tool'] }
    $rule = 'unknown'
    if ($map.Contains('rule')) { $rule = [string]$map['rule'] }
    $message = ''
    if ($map.Contains('message')) { $message = [string]$map['message'] }
    return ConvertTo-Ae2Issue -Language $language -Tool $tool -Rule $rule -Count $count -Message $message
}

function Add-StageResult {
    param(
        [Parameter(Mandatory)]$Result,
        [object]$Issues = $null,
        [object]$Metrics = $null
    )
    [void]$script:StageResults.Add($Result)
    foreach ($item in @($Issues)) {
        if ($null -eq $item) { continue }
        $issue = ConvertTo-IssueFromObject $item
        if ($null -ne $issue) {
            [void]$script:Issues.Add($issue)
        }
    }
    if ($null -ne $Metrics) {
        $map = ConvertTo-OrderedHashtable $Metrics
        foreach ($key in $map.Keys) {
            $script:Metrics[$key] = $map[$key]
        }
    }
}

function Get-OverallStatus {
    $statuses = @($script:StageResults | ForEach-Object { $_.status })
    if ($statuses -contains 'fail') { return 'fail' }
    if ($statuses -contains 'warn') { return 'warn' }
    return 'ok'
}

function Write-StageCompletion {
    param([Parameter(Mandatory)]$Result)
    $label = ([string]$Result.status).ToUpperInvariant()
    $seconds = [math]::Round(([double]$Result.duration_ms) / 1000.0, 1)
    $color = switch ([string]$Result.status) {
        'ok' { 'Green' }
        'warn' { 'Yellow' }
        'fail' { 'Red' }
        'cached' { 'Cyan' }
        'skip' { 'DarkGray' }
        default { 'White' }
    }
    Write-Host ('[{0}] {1} ({2}s)' -f $label, $Result.name, $seconds) -ForegroundColor $color
}

function Complete-Stage {
    param(
        [Parameter(Mandatory)]$Result,
        [object]$Issues = $null,
        [object]$Metrics = $null
    )
    Add-StageResult -Result $Result -Issues $Issues -Metrics $Metrics
    Write-StageCompletion -Result $Result
    if ([string]$Result.status -eq 'fail' -and $script:FailFastStages -contains [string]$Result.name) {
        throw ("Stage '{0}' failed." -f $Result.name)
    }
}

function Get-RepositoryFileList {
    $output = & git -C $script:RepoRoot ls-files --cached --others --exclude-standard 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'git ls-files failed. Cache hashing requires a git working tree.'
    }
    return @($output | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { ($_ -replace '\\', '/').Trim() } | Sort-Object -Unique)
}

function Get-PythonStageInputs {
    $all = @(Get-RepositoryFileList)
    $inputs = @($all | Where-Object { $_ -like '*.py' })
    foreach ($forced in @('pyproject.toml', 'build.py')) {
        if ($inputs -notcontains $forced) {
            $inputs += $forced
        }
    }
    return @($inputs | Sort-Object -Unique)
}

function Get-SelfCheckInputs {
    return @('run.ps1', 'build.ps1', 'build.py', 'PSScriptAnalyzerSettings.psd1', 'pyproject.toml')
}

function Get-ContentHash {
    param(
        [string[]]$RelativePaths,
        [string[]]$AdditionalValues = @()
    )
    $stream = New-Object System.IO.MemoryStream
    try {
        foreach ($rel in (@($RelativePaths) | Sort-Object)) {
            if ([string]::IsNullOrWhiteSpace($rel)) { continue }
            $pathBytes = [System.Text.Encoding]::UTF8.GetBytes([string]$rel)
            if ($pathBytes.Length -gt 0) {
                $stream.Write($pathBytes, 0, $pathBytes.Length)
            }
            $full = Join-Path $script:RepoRoot (($rel -replace '/', '\'))
            if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
                $marker = [System.Text.Encoding]::UTF8.GetBytes(('MISSING:{0}' -f $rel))
                $stream.Write($marker, 0, $marker.Length)
                continue
            }
            $fileBytes = [System.IO.File]::ReadAllBytes($full)
            if ($fileBytes.Length -gt 0) {
                $stream.Write($fileBytes, 0, $fileBytes.Length)
            }
        }
        foreach ($extra in @($AdditionalValues)) {
            $extraBytes = [System.Text.Encoding]::UTF8.GetBytes([string]$extra)
            if ($extraBytes.Length -gt 0) {
                $stream.Write($extraBytes, 0, $extraBytes.Length)
            }
        }
        $stream.Position = 0
        $hasher = [System.Security.Cryptography.SHA256]::Create()
        try {
            $hash = $hasher.ComputeHash($stream)
            return [BitConverter]::ToString($hash).Replace('-', '').ToLowerInvariant()
        }
        finally {
            $hasher.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Get-StageCacheKey {
    param(
        [Parameter(Mandatory)][string]$StageName,
        [Parameter(Mandatory)][string[]]$RelativePaths
    )
    $additional = @(
        ('schema={0}' -f $script:CacheSchemaVersion)
        ('stage={0}' -f $StageName)
        ('python={0}' -f $script:PythonVersion)
        ('ruff={0}' -f $script:RuffVersion)
    )
    return Get-ContentHash -RelativePaths $RelativePaths -AdditionalValues $additional
}

function Test-StageCache {
    param(
        [Parameter(Mandatory)][string]$StageName,
        [Parameter(Mandatory)][string]$Hash
    )
    if ($script:NoCache) { return $false }
    $hashFile = Join-Path $script:CacheDir ('{0}.sha256' -f $StageName)
    $trustFile = Join-Path $script:CacheDir ('{0}.trusted' -f $StageName)
    if (-not (Test-Path -LiteralPath $hashFile) -or -not (Test-Path -LiteralPath $trustFile)) {
        return $false
    }
    $stored = (Get-Content -LiteralPath $hashFile -Raw -Encoding utf8).Trim()
    return ($stored -eq $Hash)
}

function Write-StageCache {
    param(
        [Parameter(Mandatory)][string]$StageName,
        [Parameter(Mandatory)][string]$Hash
    )
    $hashFile = Join-Path $script:CacheDir ('{0}.sha256' -f $StageName)
    $trustFile = Join-Path $script:CacheDir ('{0}.trusted' -f $StageName)
    Set-Content -LiteralPath $hashFile -Value $Hash -Encoding utf8
    Set-Content -LiteralPath $trustFile -Value ((Get-Date).ToUniversalTime().ToString('o')) -Encoding utf8
}

function Get-LastJsonObjectLine {
    param([string[]]$Lines)
    if ($null -eq $Lines) { return $null }
    $arr = @($Lines)
    for ($i = $arr.Count - 1; $i -ge 0; $i--) {
        $line = [string]$arr[$i]
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith('{')) { return $trimmed }
    }
    return $null
}

function Get-GitMetadata {
    $head = ''
    $branch = ''
    $dirty = $false
    try {
        $head = ([string](& git -C $script:RepoRoot rev-parse HEAD 2>$null)).Trim()
        $branch = ([string](& git -C $script:RepoRoot rev-parse --abbrev-ref HEAD 2>$null)).Trim()
        $porcelain = @(& git -C $script:RepoRoot status --porcelain 2>$null)
        $dirty = @($porcelain | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count -gt 0
    }
    catch {
        $head = 'unknown'
        $branch = 'unknown'
        $dirty = $true
    }
    return [ordered]@{
        head   = $head
        branch = $branch
        dirty  = [bool]$dirty
    }
}

function Write-CiReport {
    $finished = Get-Date
    $overall = Get-OverallStatus
    $git = Get-GitMetadata
    $ciProfile = 'full'
    if ($script:Fast) { $ciProfile = 'fast' }

    $report = [ordered]@{
        schema_version  = [int]$script:ReportSchemaVersion
        started_at_utc  = $script:StartedAt.ToUniversalTime().ToString('o')
        finished_at_utc = $finished.ToUniversalTime().ToString('o')
        duration_ms     = [int]($finished - $script:StartedAt).TotalMilliseconds
        status          = $overall
        stages          = @($script:StageResults.ToArray())
        issues          = @($script:Issues.ToArray())
        metrics         = $script:Metrics
        ci              = [ordered]@{
            passed      = [bool]($overall -ne 'fail')
            profile     = $ciProfile
            skip_launch = [bool]$script:SkipLaunch
        }
        git             = $git
    }

    $json = $report | ConvertTo-Json -Depth 12
    Set-Content -LiteralPath $script:ReportPath -Value $json -Encoding utf8
}

function Write-EnforcerState {
    if (Test-Path -LiteralPath $script:ReportPath) {
        Copy-Item -LiteralPath $script:ReportPath -Destination $script:EnforcerLastCheckPath -Force
    }
    $overall = Get-OverallStatus
    $ciProfile = 'full'
    if ($script:Fast) { $ciProfile = 'fast' }
    $line = '{0} status={1} duration_ms={2} issues={3} profile={4}' -f `
    (Get-Date).ToUniversalTime().ToString('o'), `
        $overall, `
        [int]((Get-Date) - $script:StartedAt).TotalMilliseconds, `
        $script:Issues.Count, `
        $ciProfile
    Add-Content -LiteralPath $script:EnforcerStatsPath -Value $line -Encoding utf8
}

function Write-CompactSummary {
    Write-Host ''
    Write-Host '================ SUMMARY ================' -ForegroundColor Cyan
    foreach ($stage in $script:StageResults) {
        $note = ''
        if ($stage.note) { $note = '  {0}' -f $stage.note }
        Write-Host ('  {0,-18} {1,-8}{2}' -f $stage.name, $stage.status, $note)
    }
    Write-Host ('Issues: {0} total' -f $script:Issues.Count)
    Write-Host ('Report: {0}' -f $script:ReportPath)
}

function Test-PowerShellSyntax {
    param([Parameter(Mandatory)][string]$Path)
    $tokens = $null
    $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    if ($null -eq $errors) {
        return @()
    }
    return @($errors)
}

function Invoke-SelfCheckStage {
    $name = 'self-check'
    $inputs = @(Get-SelfCheckInputs)
    $cacheKey = Get-StageCacheKey -StageName $name -RelativePaths $inputs
    if (Test-StageCache -StageName $name -Hash $cacheKey) {
        Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $name -Status 'cached' -Note 'Cache hit' -DurationMs 0)
        return
    }

    $started = Get-Date
    $log = New-Object System.Collections.ArrayList
    $localIssues = New-Object System.Collections.ArrayList
    $pssaMissing = $false

    foreach ($rel in $inputs) {
        $full = Join-Path $script:RepoRoot $rel
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
            [void]$log.Add(('missing required file: {0}' -f $rel))
            [void]$localIssues.Add((ConvertTo-Ae2Issue -Language 'ci' -Tool 'self-check' -Rule 'missing_file' -Message $rel))
        }
        else {
            [void]$log.Add(('found: {0}' -f $rel))
        }
    }

    foreach ($scriptName in @('run.ps1', 'build.ps1')) {
        $full = Join-Path $script:RepoRoot $scriptName
        if (-not (Test-Path -LiteralPath $full)) { continue }
        $parseErrors = @(Test-PowerShellSyntax -Path $full)
        if (@($parseErrors).Count -gt 0) {
            foreach ($parseError in $parseErrors) {
                $msg = '{0}: {1}' -f $scriptName, $parseError.Message
                [void]$log.Add($msg)
                [void]$localIssues.Add((ConvertTo-Ae2Issue -Language 'powershell' -Tool 'parser' -Rule 'parse_error' -Message $msg))
            }
        }
        else {
            [void]$log.Add(('parser ok: {0}' -f $scriptName))
        }
    }

    $pssaModule = Get-Module -ListAvailable -Name PSScriptAnalyzer | Select-Object -First 1
    if ($null -eq $pssaModule) {
        $pssaMissing = $true
        [void]$log.Add('PSScriptAnalyzer module is missing')
    }
    else {
        Import-Module PSScriptAnalyzer -ErrorAction Stop
        $targets = @(
            (Join-Path $script:RepoRoot 'run.ps1')
            (Join-Path $script:RepoRoot 'build.ps1')
        )
        $findings = @()
        foreach ($target in $targets) {
            if (Test-Path -LiteralPath $script:PssaSettingsPath) {
                $findings += @(Invoke-ScriptAnalyzer -Path $target -Settings $script:PssaSettingsPath)
            }
            else {
                $findings += @(Invoke-ScriptAnalyzer -Path $target)
            }
        }
        [void]$log.Add(('PSScriptAnalyzer findings: {0}' -f @($findings).Count))
        foreach ($finding in $findings) {
            $msg = '{0}:{1} [{2}] {3}' -f $finding.ScriptName, $finding.Line, $finding.RuleName, $finding.Message
            [void]$log.Add($msg)
            [void]$localIssues.Add((ConvertTo-Ae2Issue -Language 'powershell' -Tool 'PSScriptAnalyzer' -Rule ([string]$finding.RuleName) -Message $msg))
        }
    }

    $pyCompile = Invoke-ExternalCommand -FilePath $script:PythonExe -ArgumentList @('-m', 'py_compile', $script:BuildPyPath)
    [void]$log.Add(('py_compile build.py exit={0}' -f $pyCompile.ExitCode))
    foreach ($line in @($pyCompile.Combined)) { [void]$log.Add([string]$line) }
    if ($pyCompile.ExitCode -ne 0) {
        [void]$localIssues.Add((ConvertTo-Ae2Issue -Language 'python' -Tool 'py_compile' -Rule 'compile_error' -Message 'build.py failed to compile'))
    }

    $tomlPath = Join-Path $script:RepoRoot 'pyproject.toml'
    $tomlCode = 'import pathlib,sys,tomllib; tomllib.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))'
    $toml = Invoke-ExternalCommand -FilePath $script:PythonExe -ArgumentList @('-c', $tomlCode, $tomlPath)
    [void]$log.Add(('tomllib pyproject.toml exit={0}' -f $toml.ExitCode))
    foreach ($line in @($toml.Combined)) { [void]$log.Add([string]$line) }
    if ($toml.ExitCode -ne 0) {
        [void]$localIssues.Add((ConvertTo-Ae2Issue -Language 'python' -Tool 'tomllib' -Rule 'parse_error' -Message 'pyproject.toml failed to parse'))
    }

    [void](Write-StageLog -StageName $name -Lines @($log))
    $duration = [int]((Get-Date) - $started).TotalMilliseconds

    if (@($localIssues).Count -gt 0) {
        Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $name -Status 'fail' -Note ('{0} finding(s)' -f @($localIssues).Count) -DurationMs $duration) -Issues $localIssues
        return
    }
    if ($pssaMissing) {
        Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $name -Status 'warn' -Note 'PSScriptAnalyzer module is missing' -DurationMs $duration)
        return
    }

    Write-StageCache -StageName $name -Hash $cacheKey
    Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $name -Status 'ok' -Note 'CI scripts validated' -DurationMs $duration)
}

function Invoke-AdapterStage {
    param(
        [Parameter(Mandatory)][string]$Name,
        [string[]]$CacheInputs = @()
    )
    $cacheable = $script:CacheableStages -contains $Name
    $mutating = $script:MutatingStages -contains $Name

    if ($cacheable -and @($CacheInputs).Count -gt 0) {
        $preHash = Get-StageCacheKey -StageName $Name -RelativePaths $CacheInputs
        if (Test-StageCache -StageName $Name -Hash $preHash) {
            Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $Name -Status 'cached' -Note 'Cache hit' -DurationMs 0)
            return
        }
    }

    $logPath = Join-Path $script:LogsDir ('{0}.log' -f $Name)
    $invoke = Invoke-ExternalCommand -FilePath $script:PythonExe -ArgumentList @(
        $script:BuildPyPath,
        '--stage', $Name,
        '--root', $script:RepoRoot,
        '--log-path', $logPath
    )

    $jsonLine = Get-LastJsonObjectLine -Lines $invoke.StdOut
    if ([string]::IsNullOrWhiteSpace($jsonLine)) {
        $note = 'Adapter JSON missing'
        if ($invoke.ExitCode -ne 0) {
            $note = 'Adapter exited {0} without JSON' -f $invoke.ExitCode
        }
        $failIssues = @(
            ConvertTo-Ae2Issue -Language 'python' -Tool 'build.py' -Rule 'adapter_error' -Message $note
        )
        Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $Name -Status 'fail' -Note $note -DurationMs $invoke.DurationMs) -Issues $failIssues
        return
    }

    $parsed = $jsonLine | ConvertFrom-Json
    $status = [string]$parsed.status
    if ($invoke.ExitCode -ne 0 -and $status -ne 'fail' -and $status -ne 'warn') {
        $status = 'fail'
    }
    $note = ''
    if ($parsed.PSObject.Properties['note']) { $note = [string]$parsed.note }
    $duration = $invoke.DurationMs
    if ($parsed.PSObject.Properties['duration_ms'] -and $null -ne $parsed.duration_ms) {
        $duration = [int]$parsed.duration_ms
    }
    $details = $null
    if ($parsed.PSObject.Properties['details']) { $details = $parsed.details }
    $result = ConvertTo-Ae2StageResult -Name $Name -Status $status -Note $note -DurationMs $duration -Details $details

    $parsedIssues = @()
    if ($parsed.PSObject.Properties['issues'] -and $null -ne $parsed.issues) {
        $parsedIssues = @($parsed.issues)
    }
    $parsedMetrics = $null
    if ($parsed.PSObject.Properties['metrics']) { $parsedMetrics = $parsed.metrics }

    if ($status -eq 'ok' -and $cacheable -and @($CacheInputs).Count -gt 0) {
        $stampInputs = $CacheInputs
        if ($mutating) {
            $stampInputs = Get-PythonStageInputs
        }
        $postHash = Get-StageCacheKey -StageName $Name -RelativePaths $stampInputs
        Write-StageCache -StageName $Name -Hash $postHash
    }

    Complete-Stage -Result $result -Issues $parsedIssues -Metrics $parsedMetrics
}

function Invoke-CodebaseMemoryStage {
    $name = 'codebase-memory'
    $started = Get-Date
    if ($script:Fast) {
        Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $name -Status 'skip' -Note 'Fast profile' -DurationMs 0)
        return
    }
    $cmd = Get-Command 'codebase-memory-mcp' -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        $duration = [int]((Get-Date) - $started).TotalMilliseconds
        Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $name -Status 'warn' -Note 'codebase-memory-mcp unavailable' -DurationMs $duration)
        return
    }
    $invoke = Invoke-ExternalCommand -FilePath $cmd.Source -ArgumentList @(
        'cli', 'index_repository', '--repo-path', $script:RepoRoot, '--mode', 'full', '--persist', 'false'
    )
    [void](Write-StageLog -StageName $name -Lines @($invoke.Combined))
    if ($invoke.ExitCode -ne 0) {
        Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $name -Status 'warn' -Note ('index_repository exited {0}' -f $invoke.ExitCode) -DurationMs $invoke.DurationMs)
        return
    }
    Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $name -Status 'ok' -Note 'Index refreshed' -DurationMs $invoke.DurationMs)
}

function Invoke-SkippedStage {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Note
    )
    Complete-Stage -Result (ConvertTo-Ae2StageResult -Name $Name -Status 'skip' -Note $Note -DurationMs 0)
}

# =============================================================================
# Pipeline
# =============================================================================
Initialize-CiDirectories
$script:PythonExe = Resolve-ProjectPython
$script:PythonVersion = Get-ToolVersionString -FilePath $script:PythonExe -ArgumentList @('-c', 'import sys; print(sys.version.split()[0])')
$script:RuffVersion = Get-ToolVersionString -FilePath $script:PythonExe -ArgumentList @('-m', 'ruff', '--version')

Write-Host ('Python: {0} ({1})' -f $script:PythonExe, $script:PythonVersion) -ForegroundColor DarkGray
Write-Host ('Ruff:   {0}' -f $script:RuffVersion) -ForegroundColor DarkGray
if ($script:Fast) {
    Write-Host 'Profile: Fast' -ForegroundColor DarkGray
}
else {
    Write-Host 'Profile: Full' -ForegroundColor DarkGray
}

$exitCode = 0
try {
    Invoke-SelfCheckStage
    $pyInputs = Get-PythonStageInputs
    Invoke-AdapterStage -Name 'fmt' -CacheInputs $pyInputs
    $pyInputs = Get-PythonStageInputs
    Invoke-AdapterStage -Name 'lint' -CacheInputs $pyInputs
    Invoke-AdapterStage -Name 'compile' -CacheInputs $pyInputs
    Invoke-AdapterStage -Name 'test'
    if ($script:Fast) {
        Invoke-SkippedStage -Name 'coverage' -Note 'Fast profile'
    }
    else {
        Invoke-AdapterStage -Name 'coverage'
    }
    Invoke-SkippedStage -Name 'security' -Note 'See CI_TODO.md (pip-audit skipped; no lock file)'
    Invoke-CodebaseMemoryStage
    if ($script:SkipLaunch) {
        Invoke-SkippedStage -Name 'launch' -Note '-SkipLaunch: launch is not applicable; scenarios start via POST /runs'
    }
    else {
        Invoke-SkippedStage -Name 'launch' -Note 'Launch is not applicable; scenarios start via POST /runs'
    }
    Invoke-SkippedStage -Name 'archive' -Note 'CI produces no release artifacts'
}
catch {
    $script:PipelineException = $_
    Write-Host ('Pipeline stopped: {0}' -f $_.Exception.Message) -ForegroundColor Red
    if ($_.ScriptStackTrace) {
        Write-Host $_.ScriptStackTrace -ForegroundColor DarkRed
    }
    $exitCode = 1
}
finally {
    Write-CiReport
    Write-EnforcerState
    Write-CompactSummary
}

if ((Get-OverallStatus) -eq 'fail' -or $null -ne $script:PipelineException) {
    $exitCode = 1
}
exit $exitCode
