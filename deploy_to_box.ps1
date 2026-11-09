# 把验证录制的脚本部署到 Box 的 ~/ROBOT/fr5_verify（不碰 ~/ROBOT/fr5_record）。
# 用法（在本文件夹打开 PowerShell）：  .\deploy_to_box.ps1
# 手眼外参、起点、速度标定和拖动标记第一次从 fr5_record 拷过来，之后不覆盖；waypoint_corrections.json 有就更新。
param([string]$BoxHost = "user@10.255.250.132")

$root = $PSScriptRoot
$inputs = "handeye_s003_result.json", "start_pose.json", "speed_calibration.json", "limit_poses.json"

ssh $BoxHost "mkdir -p ~/ROBOT/fr5_verify ~/ROBOT/sessions_verify"
if ($LASTEXITCODE -ne 0) { Write-Error "连不上 Box（$BoxHost）"; exit 1 }

$files = Get-ChildItem "$root\scripts" -File | Where-Object { $inputs -notcontains $_.Name }
scp $files.FullName "${BoxHost}:ROBOT/fr5_verify/"
if ($LASTEXITCODE -ne 0) { Write-Error "拷贝脚本失败"; exit 1 }

ssh $BoxHost "cd ~/ROBOT && for f in $($inputs -join ' '); do cp -n fr5_record/`$f fr5_verify/; done; sha256sum fr5_verify/handeye_s003_result.json; ls -l fr5_verify/*.json"
if ($LASTEXITCODE -ne 0) { Write-Error "拷贝输入文件失败"; exit 1 }

Write-Output "已部署到 ${BoxHost}:ROBOT/fr5_verify"
