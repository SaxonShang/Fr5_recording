# 从 Box 拉取数据录制脚本的最新版本到本地备份（只拉代码和配置，不拉录制数据）。
# 用法（在本文件夹打开 PowerShell）：  .\sync_from_box.ps1
param([string]$BoxHost = "user@10.255.250.132")

$root = $PSScriptRoot
New-Item -ItemType Directory -Force "$root\scripts", "$root\zed_record" | Out-Null

# Python 运行时可能留下 __pycache__，先清掉，scp 才能用通配符只拷文件
ssh $BoxHost "find ~/ROBOT/fr5_record -name __pycache__ -prune -exec rm -rf {} +"
scp "${BoxHost}:ROBOT/fr5_record/*" "$root\scripts\"
if ($LASTEXITCODE -ne 0) { Write-Error "拷贝 fr5_record 失败"; exit 1 }

ssh $BoxHost "cd ~/ROBOT/zed_imu && diff -u zed_record.cpp.bak-20261008 zed_record.cpp > /tmp/zed_record.patch; true"
scp "${BoxHost}:ROBOT/zed_imu/zed_record.cpp" "${BoxHost}:/tmp/zed_record.patch" "$root\zed_record\"
if ($LASTEXITCODE -ne 0) { Write-Error "拷贝 zed_record 失败"; exit 1 }
ssh $BoxHost "rm -f /tmp/zed_record.patch"

Write-Output "已同步到 $root"
