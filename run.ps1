# =============================================================================
# run.ps1 — Thin wrapper (entry point)
# =============================================================================
# Validates flags and forwards execution to build.ps1. Does not implement stages.
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Fast,
    [switch]$SkipLaunch,
    [Alias('NoCashe')]
    [switch]$NoCache,
    [switch]$ForceAll,
    [switch]$Clean,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Show-Help {
    Write-Host 'MerchantBench local CI' -ForegroundColor Cyan
    Write-Host ''
    Write-Host 'Usage: .\run.ps1 [options]'
    Write-Host ''
    Write-Host 'Profiles:'
    Write-Host '  -Fast          Skip coverage and codebase-memory'
    Write-Host ''
    Write-Host 'Flags:'
    Write-Host '  -SkipLaunch    Record an explicit launch skip (launch is always skipped here)'
    Write-Host '  -NoCache       Ignore cache hits; rewrite trust stamps on success'
    Write-Host '  -ForceAll      Same as -NoCache for this pipeline'
    Write-Host '  -Clean         Delete .ci_cache/ before running'
    Write-Host '  -Help          Show this help'
}

if ($Help) {
    Show-Help
    exit 0
}

if ($null -ne $RemainingArguments -and @($RemainingArguments).Count -gt 0) {
    $unknown = ($RemainingArguments -join ', ')
    Write-Host ('Error: Unknown argument(s): {0}' -f $unknown) -ForegroundColor Red
    Write-Host 'Valid parameters: -Fast, -SkipLaunch, -NoCache, -ForceAll, -Clean, -Help' -ForegroundColor Yellow
    exit 1
}

$forward = @{}
foreach ($entry in $PSBoundParameters.GetEnumerator()) {
    if ($entry.Key -eq 'Help' -or $entry.Key -eq 'RemainingArguments') {
        continue
    }
    $forward[$entry.Key] = $entry.Value
}

if ($ForceAll) {
    $forward['NoCache'] = $true
}

$buildScript = Join-Path $PSScriptRoot 'build.ps1'
& $buildScript @forward
$exitCode = $LASTEXITCODE
if ($null -eq $exitCode) {
    $exitCode = 0
}
exit $exitCode
