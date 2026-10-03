param(
    [ValidateRange(0, 14)][int]$MaxParallel = 0,
    [switch]$SkipBenchmark
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot
$Logs = Join-Path $ProjectRoot 'data\logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Choose([int]$N, [int]$K) {
    $K = [Math]::Min($K, $N - $K)
    $Value = 1.0
    for ($I = 1; $I -le $K; $I++) { $Value *= ($N - $K + $I) / $I }
    return $Value
}

function CandidateEntries([int]$N) {
    $K = [int][Math]::Floor($N / 2)
    return (Choose $N $K) + 2 * $N * (Choose ($N - 2) ($K - 1))
}

$KnownPeaks = @{ 16 = 0.154; 18 = 0.173; 20 = 0.249; 22 = 0.538; 24 = 1.784; 26 = 6.796 }
$BaseEntries = CandidateEntries 26
function EstimateGiB([int]$N) {
    if ($KnownPeaks.ContainsKey($N)) { return [double]$KnownPeaks[$N] }
    $Scaled = 1.25 * 6.796 * (CandidateEntries $N) / $BaseEntries
    return [Math]::Max(0.25, $Scaled)
}

$DockerBytes = & docker info --format '{{.MemTotal}}'
if ($LASTEXITCODE -ne 0) { throw 'Docker Engine is unavailable.' }
$MemoryGiB = [double]($DockerBytes | Select-Object -Last 1) / 1GB
$MemoryBudget = 0.70 * $MemoryGiB
Write-Host ("Docker memory: {0:N2} GiB; job budget: {1:N2} GiB; CPU cap: 14." -f $MemoryGiB, $MemoryBudget)
& docker compose build notebook
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed.' }

function Run-Batch($Cases, [int]$Limit, [double]$Budget, [string]$Label) {
    $Pending = @($Cases | Sort-Object -Property N -Descending)
    $Running = @()
    $Failed = @()
    $Completed = 0
    $StartedAt = Get-Date
    while ($Pending.Count -gt 0 -or $Running.Count -gt 0) {
        $StartedSomething = $false
        while ($Running.Count -lt $Limit -and $Pending.Count -gt 0) {
            $Used = 0.0
            foreach ($Job in $Running) { $Used += $Job.MemoryGiB }
            $Chosen = $null
            foreach ($Candidate in $Pending) {
                $Need = (EstimateGiB $Candidate.N) + 0.5
                $Already26 = @($Running | Where-Object { $_.N -eq 26 }).Count -gt 0
                if (($Used + $Need -le $Budget -or $Running.Count -eq 0) -and
                    -not ($Candidate.N -eq 26 -and $Already26)) {
                    $Chosen = $Candidate
                    break
                }
            }
            if ($null -eq $Chosen) { break }
            $Pending = @($Pending | Where-Object { $_.Id -ne $Chosen.Id })
            $Out = Join-Path $Logs ($Chosen.Id + '.out.log')
            $Err = Join-Path $Logs ($Chosen.Id + '.err.log')
            $Args = @('compose', 'run', '--rm', '-T', 'notebook', 'python', 'scripts/run_case.py',
                      [string]$Chosen.N, [string]$Chosen.Boundary)
            $Process = Start-Process -FilePath 'docker' -ArgumentList $Args -WorkingDirectory $ProjectRoot `
                -RedirectStandardOutput $Out -RedirectStandardError $Err -WindowStyle Hidden -PassThru
            $Running += [pscustomobject]@{ Process = $Process; N = $Chosen.N; Boundary = $Chosen.Boundary;
                Id = $Chosen.Id; MemoryGiB = (EstimateGiB $Chosen.N) + 0.5; ErrorLog = $Err }
            $StartedSomething = $true
        }
        $StillRunning = @()
        foreach ($Job in $Running) {
            $Job.Process.Refresh()
            if ($Job.Process.HasExited) {
                $Completed++
                if ($Job.Process.ExitCode -ne 0) {
                    Write-Warning ("{0} failed (exit {1}): {2}" -f $Job.Id, $Job.Process.ExitCode, $Job.ErrorLog)
                    $Failed += $Job
                }
                Write-Progress -Activity $Label -Status "$Completed / $($Cases.Count)" `
                    -PercentComplete (100 * $Completed / $Cases.Count)
            } else { $StillRunning += $Job }
        }
        $Running = $StillRunning
        if ($Running.Count -gt 0 -or (-not $StartedSomething -and $Pending.Count -gt 0)) {
            Start-Sleep -Milliseconds 400
        }
    }
    Write-Progress -Activity $Label -Completed
    foreach ($Job in $Failed) {
        Write-Host "Retrying $($Job.Id) alone."
        & docker compose run --rm -T notebook python scripts/run_case.py $Job.N $Job.Boundary
        if ($LASTEXITCODE -ne 0) { throw "Case $($Job.Id) failed even when run alone." }
    }
    return ((Get-Date) - $StartedAt).TotalSeconds
}

$AllCases = @(foreach ($N in 3..26) {
    foreach ($Boundary in @('OBC', 'PBC')) {
        [pscustomobject]@{ N = $N; Boundary = $Boundary; Id = ('N{0:D2}_{1}' -f $N, $Boundary) }
    }
})
$ChosenLimit = $MaxParallel
$Benchmarks = @()
if ($ChosenLimit -eq 0 -and -not $SkipBenchmark) {
    $Sample = @($AllCases | Where-Object { $_.N -ge 17 -and $_.N -le 22 })
    $BestSeconds = [double]::PositiveInfinity
    $ChosenLimit = 1
    foreach ($Limit in @(1, 2, 4, 8, 14)) {
        $Elapsed = Run-Batch $Sample $Limit $MemoryBudget "Benchmark: $Limit containers"
        $Benchmarks += [pscustomobject]@{ containers = $Limit; wall_seconds = $Elapsed }
        Write-Host ("Benchmark cap {0,2} containers: {1:N2} s" -f $Limit, $Elapsed)
        if ($Elapsed -lt 0.95 * $BestSeconds) { $BestSeconds = $Elapsed; $ChosenLimit = $Limit }
    }
} elseif ($ChosenLimit -eq 0) { $ChosenLimit = 1 }

Write-Host "Selected parallel limit: $ChosenLimit"
if ($Benchmarks.Count -gt 0) {
    [pscustomobject]@{ docker_memory_gib = $MemoryGiB; memory_budget_gib = $MemoryBudget;
        chosen_containers = $ChosenLimit; sample_sizes = @(17, 18, 19, 20, 21, 22);
        measurements = $Benchmarks } | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $ProjectRoot 'data\benchmark.json') -Encoding utf8
}
$SampleIds = if ($MaxParallel -eq 0 -and -not $SkipBenchmark) {
    @($AllCases | Where-Object { $_.N -ge 17 -and $_.N -le 22 } | ForEach-Object { $_.Id })
} else { @() }
$Remaining = @($AllCases | Where-Object { $_.Id -notin $SampleIds })
$Elapsed = Run-Batch $Remaining $ChosenLimit $MemoryBudget 'Full XXX grid'
Write-Host ("Completed all 48 cases. Remaining-batch time: {0:N2} s" -f $Elapsed)
