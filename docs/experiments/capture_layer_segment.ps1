param([Parameter(Mandatory=$true)][string]$Name,[Parameter(Mandatory=$true)][int]$Launcher)
$ErrorActionPreference='Stop'
if($Name -notmatch '^m2-stream-layer-(evidence|control)768-s[1-3]$'){throw 'owned segment name required'}
$pipe='runs/'+$Name+'-pipeline';$run='runs/'+$Name;$out='docs/evidence/'+$Name
$parent=Get-Content ($pipe+'-job.json') -Raw|ConvertFrom-Json
$child=Get-Content ($run+'-job.json') -Raw|ConvertFrom-Json
$cur=Get-Content ($pipe+'/curation-job.json') -Raw|ConvertFrom-Json
$result=Get-Content ($pipe+'/result.json') -Raw|ConvertFrom-Json
$summary=Get-Content ($run+'/summary.json') -Raw|ConvertFrom-Json
$progress=Get-Content ($run+'/progress.json') -Raw|ConvertFrom-Json
$scope=Get-Content ($out+'/scope.json') -Raw|ConvertFrom-Json
$enabled=$Name -match '-evidence768-'
$partial=$summary.steps -lt 768
$expectedExit=if($partial){2}else{0}
$expectedStatus=if($partial){'step_partial'}else{'complete'}
if($parent.status -ne 'passed' -or $cur.status -ne 'passed' -or $child.exit_code -ne $expectedExit -or
 $result.training_status -ne $expectedStatus -or $summary.status -ne $expectedStatus -or
 $result.global_steps -ne $summary.steps -or $scope.steps -ne $summary.steps -or
 $result.target_steps -ne 768 -or $summary.requested_steps -ne 768 -or
 $result.layer_evidence -ne $enabled -or $summary.config.layer_evidence -ne $enabled -or
 $scope.weights_sha256 -ne $summary.checkpoint_sha256 -or
 $scope.source_catalogue_sha256 -ne $summary.catalogue_sha256 -or
 $result.controller_sha256 -ne (Get-FileHash scripts/run_stream_segment.py).Hash.ToLower()) {throw 'segment identity/status'}
if((Get-FileHash ($run+'/checkpoint/weights.pt')).Hash.ToLower() -ne $summary.checkpoint_sha256){throw 'actual weight hash'}
$ptr=Get-Content ($run+'/latest-training.json') -Raw|ConvertFrom-Json
if($ptr.generation -notmatch '^step-[0-9]{6}-[0-9a-f]{32}$'){throw 'training generation'}
$snapshot=$run+'/training-snapshots/'+$ptr.generation
if((Get-FileHash ($snapshot+'/manifest.json')).Hash.ToLower() -ne $ptr.manifest_sha256){throw 'training manifest hash'}
$training=Get-Content ($snapshot+'/manifest.json') -Raw|ConvertFrom-Json -AsHashtable
if($training.step -ne $summary.steps -or (Get-FileHash ($snapshot+'/tensors.pt')).Hash.ToLower() -ne $training.tensors_sha256){throw 'training payload hash'}
$ids=@($Launcher,$parent.supervisor_pid,$parent.child_pid,$child.supervisor_pid,$child.child_pid,$progress.pid,$cur.supervisor_pid,$cur.child_pid)
foreach($pidValue in $ids){if(Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue"){throw "PID live $pidValue"}}
if(Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python' -and $_.CommandLine -match $Name}){throw 'matching command live'}
$copies=@{($pipe+'-job.json')='controller-job.json';($pipe+'.log')='controller.log';($pipe+'.stderr.log')='controller.stderr.log';($pipe+'/curation-job.json')='controller-curation-job.json';($pipe+'/curation.log')='controller-curation.log';($pipe+'/curation.stderr.log')='controller-curation.stderr.log';($pipe+'/result.json')='controller-result.json';$PSCommandPath='controller-collector.ps1'}
$hashes=@{}
foreach($from in $copies.Keys){$to=$out+'/'+$copies[$from];Copy-Item $from $to;$h=(Get-FileHash $from).Hash.ToLower();if($h -ne (Get-FileHash $to).Hash.ToLower()){throw 'copy hash'};$hashes[$copies[$from]]=$h}
$proof=@{status=$scope.status;global_steps=$summary.steps;requested_steps=768;new_updates=$summary.updates_this_run;
 layer_evidence=$enabled;controller_sha256=$result.controller_sha256;weights_sha256=$result.weights_sha256;
 snapshot_manifest_sha256=$ptr.manifest_sha256;snapshot_payload_sha256=$training.tensors_sha256;
 verified_dead_pids=$ids;artifact_sha256=$hashes;
 parent_seconds=$parent.elapsed_seconds;parent_peak_bytes=$parent.peak_job_commit_bytes;
 child_seconds=$child.elapsed_seconds;child_peak_bytes=$child.peak_job_commit_bytes;
 curator_seconds=$cur.elapsed_seconds;curator_peak_bytes=$cur.peak_job_commit_bytes}
$proof|ConvertTo-Json -Depth 5|Set-Content ($out+'/controller-receipt-scope.json')
$proof|Select-Object status,global_steps,new_updates,parent_seconds,parent_peak_bytes|ConvertTo-Json -Compress
