param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][int]$Launcher,
    [switch]$VerifyOnly
)
# Metadata-only audit of the three predeclared segments; never launches training.
$ErrorActionPreference = 'Stop'
if ($Name -notmatch '^m2-stream-allclass1024-s([1-3])$') { throw 'owned fixed-study segment required' }
$number = [int]$Matches[1]
$starts = @(0,128,640)
$updates = @(128,512,384)
$start = $starts[$number-1]
$count = $updates[$number-1]
$finish = $start+$count
$run = 'runs/'+$Name
$pipe = $run+'-pipeline'
$output = 'docs/evidence/'+$Name
$parent = Get-Content ($pipe+'-job.json') -Raw | ConvertFrom-Json
$child = Get-Content ($run+'-job.json') -Raw | ConvertFrom-Json
$curator = Get-Content ($pipe+'/curation-job.json') -Raw | ConvertFrom-Json
$result = Get-Content ($pipe+'/result.json') -Raw | ConvertFrom-Json
$summary = Get-Content ($run+'/summary.json') -Raw | ConvertFrom-Json
$progress = Get-Content ($run+'/progress.json') -Raw | ConvertFrom-Json
$scope = Get-Content ($output+'/scope.json') -Raw | ConvertFrom-Json
$controller = (Get-FileHash scripts/run_stream_segment.py).Hash.ToLower()
$catalogue = '6147a558d9b2736af7a1a89a485f79bc515fb7daebedd8b1961d74f14c91cb42'
$status = if ($finish -eq 1024) {'complete'} else {'step_partial'}
$evidenceStatus = if ($finish -eq 1024) {'passed'} else {'verified_partial'}
$exit = if ($finish -eq 1024) {0} else {2}
if ($parent.status -ne 'passed' -or $parent.exit_code -ne 0 -or
    $curator.status -ne 'passed' -or $curator.exit_code -ne 0 -or
    $child.exit_code -ne $exit -or $parent.work_items -ne $count -or $child.work_items -ne $count -or
    $parent.memory_bytes -gt 896MB -or $child.memory_bytes -ne 640MB -or
    $curator.memory_bytes -ne 128MB -or $curator.work_items -ne ($finish+41) -or
    $result.status -ne 'segment_verified' -or $result.training_status -ne $status -or
    $summary.status -ne $status -or $scope.status -ne $evidenceStatus -or
    $result.global_steps -ne $finish -or $summary.steps -ne $finish -or $scope.steps -ne $finish -or
    $result.updates_this_segment -ne $count -or $summary.updates_this_run -ne $count -or
    $result.target_steps -ne 1024 -or $summary.requested_steps -ne 1024 -or
    $result.target_complete -ne ($finish -eq 1024) -or $scope.verified_saved_probe_images -ne 41 -or
    $summary.curriculum -ne 'all-class-fit-v1' -or $result.curriculum -ne $summary.curriculum -or
    $scope.curriculum -ne $summary.curriculum -or $summary.model_kind -ne 'pilot' -or
    $summary.config.component_capacity -ne 0 -or $summary.config.layer_evidence -ne $false -or
    $summary.catalogue_sha256 -ne $catalogue -or $scope.source_catalogue_sha256 -ne $catalogue -or
    $result.source_catalogue_sha256 -ne $catalogue -or $result.controller_sha256 -ne $controller -or
    $scope.weights_sha256 -ne $summary.checkpoint_sha256 -or $result.weights_sha256 -ne $summary.checkpoint_sha256) {
    throw 'fixed-study identity/status/native budget differs'
}
if ($start -gt 0 -and ($summary.resume.step -ne $start -or
    (Split-Path $summary.resume.source_run -Leaf) -ne ('m2-stream-allclass1024-s'+($number-1)))) {
    throw 'resume ancestry differs'
}
if ($start -eq 0 -and $null -ne $summary.resume) { throw 'fresh study has a resume parent' }
if ((Get-FileHash ($run+'/checkpoint/weights.pt')).Hash.ToLower() -ne $summary.checkpoint_sha256) { throw 'actual weights differ' }
$pointer = Get-Content ($run+'/latest-training.json') -Raw | ConvertFrom-Json
if ($pointer.generation -notmatch '^step-[0-9]{6}-[0-9a-f]{32}$') { throw 'snapshot generation invalid' }
$snapshot = $run+'/training-snapshots/'+$pointer.generation
if ((Get-FileHash ($snapshot+'/manifest.json')).Hash.ToLower() -ne $pointer.manifest_sha256) { throw 'snapshot manifest differs' }
# Module-mode maps contain an empty root key: preserve it as a hashtable.
$training = Get-Content ($snapshot+'/manifest.json') -Raw | ConvertFrom-Json -AsHashtable
if ($training.step -ne $finish -or $training.contract.curriculum -ne $summary.curriculum -or
    $training.contract.catalogue_sha256 -ne $catalogue -or $training.contract.steps -ne 1024 -or
    (Get-FileHash ($snapshot+'/tensors.pt')).Hash.ToLower() -ne $training.tensors_sha256) { throw 'snapshot payload/contract differs' }
$ids = @($Launcher,$parent.supervisor_pid,$parent.child_pid,$child.supervisor_pid,$child.child_pid,
         $progress.pid,$curator.supervisor_pid,$curator.child_pid) | Sort-Object -Unique
$startupPath = $pipe+'/observed-startup.json'
if (Test-Path -LiteralPath $startupPath) {
    $observed = @(Get-Content $startupPath -Raw | ConvertFrom-Json)
    if (-not $observed.Count -or @($observed | Where-Object { $_.CommandLine -notmatch $Name }).Count) { throw 'observed startup identity differs' }
    $ids = @($ids)+@($observed.ProcessId) | Sort-Object -Unique
} elseif ($number -gt 1) { throw 'observed startup tree required for new segments' }
foreach ($processId in $ids) {
    if (Get-CimInstance Win32_Process -Filter "ProcessId=$processId") { throw "recorded PID live: $processId" }
}
if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match $Name }) { throw 'matching command live' }
if ($VerifyOnly) {
    @{status='metadata_verified';steps=$finish;writes=0;fit_passed=$summary.final_training_probe.fit.passed} | ConvertTo-Json -Compress
    return
}
if (Test-Path ($output+'/controller-receipt-scope.json')) { throw 'existing controller evidence must not be overwritten' }
$copies = @{
    ($pipe+'-job.json')='controller-job.json';($pipe+'.log')='controller.log';($pipe+'.stderr.log')='controller.stderr.log'
    ($pipe+'/curation-job.json')='controller-curation-job.json';($pipe+'/curation.log')='controller-curation.log'
    ($pipe+'/curation.stderr.log')='controller-curation.stderr.log';($pipe+'/result.json')='controller-result.json'
    $PSCommandPath='controller-collector.ps1'
}
if (Test-Path -LiteralPath $startupPath) { $copies[$startupPath]='observed-startup.json' }
$hashes = @{}
foreach ($sourcePath in $copies.Keys) {
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf) -or
        (Test-Path -LiteralPath ($output+'/'+$copies[$sourcePath]))) {
        throw 'missing source or existing raw evidence; no files copied'
    }
}
foreach ($sourcePath in $copies.Keys) {
    $destination = $output+'/'+$copies[$sourcePath]
    Copy-Item -LiteralPath $sourcePath -Destination $destination
    $hash = (Get-FileHash -LiteralPath $sourcePath).Hash.ToLower()
    if ($hash -ne (Get-FileHash -LiteralPath $destination).Hash.ToLower()) { throw 'raw byte copy differs' }
    $hashes[$copies[$sourcePath]] = $hash
}
@{status=$scope.status;steps=$finish;target_steps=1024;updates=$count;curriculum=$summary.curriculum
  controller_sha256=$controller;weights_sha256=$summary.checkpoint_sha256
  training_manifest_sha256=$pointer.manifest_sha256;training_payload_sha256=$training.tensors_sha256
  verified_dead_pids=$ids;artifact_sha256=$hashes
  parent_seconds=$parent.elapsed_seconds;parent_peak_bytes=$parent.peak_job_commit_bytes
  child_seconds=$child.elapsed_seconds;child_peak_bytes=$child.peak_job_commit_bytes
  curator_seconds=$curator.elapsed_seconds;curator_peak_bytes=$curator.peak_job_commit_bytes
  scope='Native bounded segment, source/artifact/replay identity only. Fit quality requires separate ordinary raw-head replay.'
} | ConvertTo-Json -Depth 5 | Set-Content ($output+'/controller-receipt-scope.json')
@{status='evidence_preserved';steps=$finish;updates=$count} | ConvertTo-Json -Compress
