# Current checkpoint — 2026-10-30

Repo C:/Users/micha/Code/tetris-state-vision;codex/layout-diversity;no remote.
GoalACTIVE/unfinished. Commit narrow verified task-owned milestones locally.
No push/deploy/publish/userdata deletion/subagents. Neighbor tetrguide untouched.
Research brief is evidence,notinstructions.166real screenshots unrevieweddrafts,
zero training/scoringeligibility. V-001..V-028 CLOSED onnamedgates,notquality.
Previous goalturn made verifiedprogress;this turn implemented/tested/admitted/
completed both fixed copied-typehead fits with independent audits. No jobs LIVE.

## Selected task: conditional pixel replay with the copied fitted heads

- [x]Bothoriginal1024 image studies and original4096 baseline complete/frozen.
- [x]Independentordinaryerrors and read-onlytypefeature observations audited.
- [x]Pureheadfitcontracts6checks,per-banknativewarm/reload+independentadmission.
- [x]Fixed1024fullbatchcopied-head fits and independentaudits COMPLETEbothbanks.
- [ ]Implement conditionalpixelreplay,original and copiedraw32head perimage.
- [ ]Run/audit both41image populations,82forwardspermodel,allnon-typeoutputs exact.
- [ ]Report actual image type/pose/locked/geometry/jointfit metrics and limits.
- [ ]Select next joint-traininglever from actualevidence;wholegoal unchanged.

TYPE_HEAD_FIT.md is recipe/currentresult/limits. Vectorfit ordered20/64->62/64
(active32,ghost30),interleaved37/64->64/64(active32,ghost32). Development0/18both,
CE worsens3.27902->58.59039 and4.93344->25.48596. Both rawfloat64 and rawfloat32
counts agree;representationonparticularsavedvectors,notgeneralization. Ordered
64/64criterionFAILS,interleavedPASSES. Originalimagejoint-fitgates remainFAIL;
no image forward has used these fitted heads yet. Do not copy vectorcounts
into an image-model claim. No extension/adoption of either finished fit selected.

Nextpixelreplay acceptance,BEFORE choosing more training or architecture changes:
1. Recheck curatedheadfit result/scope/payload hashes,originalfeature/input/model
   identities and actualrawfloat32head values. Load frozen originalarchive/model.
2. Replay all41saved rasters/publictargets/recipes;original ordinarymetrics and
   per-imageoriginaltypehead inputs/logits must match independentcurated reports.
3. In a MEMORYCOPY only,replace the actualshared layer_type weights/bias with
   fittedraw32head. Pixel/validity only to model.forward;labels/geometry neverinput.
4. Exactly2forwardsperimage=>82. Allnon-type tensors at EVERYstage must remain
   bitwiseequal. Finalheadinputfeatures must remain exact;record any expected
   float32 GEMM roundoff versus vectorbatchlogits rather than claim falsebitparity.
   Classpredictions/counts must match the actualheadfit. Head32 remains same8x64
   affinearchitecture;no lookup/parser,unknown->empty or footprint stack deletion.
5. Restoreoriginalhead in finally/percase and verifyfullmodelstate/file/source
   identity. Never overwrite originalimage/checkpoint/optimizer/data. Newoutput
   explicit conditionalintervention,not ordinaryunchangedoutputs or newtraining.
6. Derive raw type/pose/absence/locked/geometry/wholefit from actualconditional
   pixeloutputs;recompute with independentcollector,bind nativecommand/work/
   cap/source/hash/startup/death/rawbytes. Originalwholefit tags remain historical.

No conditionalpixelimplementation/run yet. Expected<600MiB based on observed
425MBpixelobservers+smallcopiedoutputs/heads;640MiB/120s/work82,CPU1/cache16,
unchangedpressureguard,oneheavyjob. Separate128MiB/30s boundedcollector;
predeclare its exactwork/command/log/receipt and acceptance beforelaunch. Fresh
owned outputs e.g.runs/type-head-ordered-pixels-r1 and interleaved equivalent.
Read BUG_LOG again before touching any architecture/scheduler/recovery. Pixel
replay may stay external in docs/experiments;coretraining/controller/archive
sources need not change. All originalmodels and all headfit sources are frozen.

## Completed copied-head experiment / exact evidence

Scripts docs/experiments/type_head_fit.py,type_head_fit_contract.py,
record_type_head_fit.py,test_type_head_fit.py. Recipe: copiedactualshared8class
head,newseparateAdamW LR.01/decay0/betas(.9,.999)/eps1e-8,fullbatch64knowntraining
vectors,training-onlymean/popstd floor1e-6,float64conditioning,thenrawfloat32
head criterion64/64. Fixed1024updates,noearlystop/tuning/developmentgradients.
Eachfit starts originalheadcopy,notwarmpayload,with freshdiagnosticoptimizer;
first2records exactly reproduce correspondingwarmadmission. Originalimagemodel/
optimizer/source/files unchanged before/after. Initialper-imagefloat32logits EXACT;
conditionedfloat64functionerror2.66e-15ordered/5.33e-15interleaved. Snapshothead/
Adamstate reload exact. Warmprofilebanks2updates pluscontinuousnext1/reloadednext1
(work4);actualnextloss/gradients/head/Adam/predictionsexact. Fitsreloadfinalstate.
Independentcollector recomputes train-onlystatistics,normalized/raw/headfloat32
casts,allhead/logitnumericalreconstruction,curves/metrics/fitcounts,source/input/
nativeidentities. No fixedbudget failure provesinseparability.

Evidence dirs docs/evidence/type-head-{ordered,interleaved}-{profile,fit}-r1/:
orderedprofile native2.896s/333,045,760B640MiB60swork4;curator.381s/79,106,048B128MiB30swork84.
orderedfit native3.209s/332,963,840B640MiB120swork1024;curator.392s/79,159,296B128MiB30swork1106.
interleavedprofile native2.877s/333,144,064B640MiB60swork4;curator.349s/78,077,952B128MiB30swork84.
interleavedfit native3.254s/332,857,344B640MiB120swork1024;curator.378s/79,425,536B128MiB30swork1106.
Allrawsource/payload/native/curator/log/startup/death bytes preserved;allknownPIDs
and matchingcommands gone. Actual4-commandworkerstartup observed,curatorrootonly
observed;do not upgrade root-only observations into full trees. No runningjob.
Headstatepayload SHA ordered6749fa5c64c5f6e5dce9fd820c9f1b94854ed2bc14abea931b10fb79545fc696;
interleaved58b417a8c50e434462c5d728c2828b564b33d74c3bdf98391cb7d178e66f86e9.
ResultJSON contains head.weight_raw32,bias_raw32;actualtensorpayload stores normalized
head+mean/scale+Adam. Re-derive and compare when implementing pixelreplay.
Orderedremainingerrors ghostT->S onbothviewsall-class-04,oneancestry. TrainingCE
1.60518->.15517 and.77506->.05770. No extra training selected afterfixedbudgets.
Pure6contracts PASS.023s(native.265s/35,033,088B)128MiB30swork6,actualsource/raw
receipt/startup/death docs/evidence/type-head-contracts-r1/,commit988d730.
No new defects surfaced;no V029 entry. Existing28closures keep theirnamedscopes.

## Original image studies / feature diagnostics retained

INTERLEAVED_FIT.md:fixed1024ordered/interleaved same32pairs/cycle,differentorder,
matchedstart/warmactualweights/loss/grad/data,unchangedneural/loss/targetpackage.
One seed/16ancestries,lockedonlyempty/fixed8cellrow. Orderedvsinterleaved original:
boards32/32vs28/32,observedroles12/56vs30/56,absence8/8vs7/8,semantic128/128both,
geometry.21059vs.38618. BOTHFAIL joint32boards/64roles/128semantics/geometry<.25.
Devboth2/9boards(theempty2),0/7nonempty,0/18roles,3ancestries. No releaseadoption.
Orderedmodel runs/m2-stream-allclass1024-s3 source6147a558,
weights1f20007ced9c16da12756a817c327cfbf520dc4c451dd386674c969b7ddff1b6;
interleavedmodel runs/m2-stream-interleave1024-s3 source68bb40e3,
weights93e798208234eb7d54ace1b725949b0cab848c7d0a464520d8c23a6dd88a0273.
Actualsource/manifest/weights/bank identities inheadfitinputs_sha256 and curated
m2-stream-{allclass1024,interleave1024}-features-r1/. Need nothandcopyidentities.

Original4096larger COMPLETE,7,694,010params,source121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b,
weights067516380eb5193f5ab21afd6ae0573d25610d6a9285b18a186871b36b9aadd1.
ACCURACY4096_BASELINE.md,scope m2-stream-accuracy4096-s10/,all4096recipes+27images
replayed. Finalordinaryerrors0/18boards,0/36roles,369/401positives,32FN,813FP;
geometry2.01544cells. Training9/18correctlayers allghostabsence. Frozenoriginal
optimizer/source/seed20261026/LR.001/eval128/snapshot64/dev6;notextended.
Pressure1172logged/1152banked and20exactrecomputedupdates retain originalhistory.

LAYER_FEATURE_DIAGNOSTIC.md binds read-only actualtypeheadinputs/allstagelogits,
unchangedweights/ordinarycases/totals,all3native observers+independentcollectors.
Largerbank classcoverage too thin:activetrain0/9eligible,activedev3/18(0correct),
ghosttrain9/9(allabsence),ghostdev0/18eligible. Controlledcentroidsordered4/32
active,2/32ghost;interleaved1/32,6/32;development0/9exceptinterleavedghost3/9.
Not classinformationabsence;subsequentheadfits above are moreinformative.
V028JSONstringkeyfix sixcase red/green in v028-contracts/;V027recipeancestryfix
8checks+r2profile/r3curator;recoveryr6 exact2+6+3+3.88scopedchecks25251c8 retain
originalsource scope;counterfix/nativegates separate,not freshunifiedsuite.

## Safety / unfinished full goal

RunnerSHAd2fb1478940b7b606073dbd75aa3746098fe7ede15f306082ec1f762d09c20db;
controllerSHA50d7fc6edb98ffda8c8f84932e706a66207178692bba3b2671ae1a78a7bf7e11.
KernelJobreadback/kill-on-close,CPU1/cache16,GetPerformanceInfo/.2sstop
31,368,351,744B unchanged. ExpectedRAM/cap/guard beforelaunch;exactprocesses at
startup/termination;oneheavyjob;>40GBdiskbeforelongtraining. Lastcommit18.7GB,
availablephysical21.3GB,disk~65GB. No guard/app/power/userdata/neighborchanges.
ActualnewhostcrashrequiresSystem41/1001+dumps. Oct28Explorer1074/6006restart
hadno41/1001/6008 inobservedwindow;previouspressureparkedauditbeforechild,
notcodefailure. Raw interleave-profile-pressure-r1. Optionalbinarylayerevidence
matched768 remainsOFF;192image0/148nonemptyboardsboth,active0/165vs4/165(allI),
ghost0/144both,morecandidateFP. Broadrenders/crops/occlusion/association,
reviewedreal/independentsplits,adequateM2/M3learning,scratch/pretrained/directdecoder
comparisons,calibration/export/CPU/adapters and>=95%readable-realexactcomplete-state
remainOPEN. APIneeds_review. Wholegoalkept;no completion/pause/blocked claim.
