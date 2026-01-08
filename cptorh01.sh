#!/bin/bash

# 创建目标目录
mkdir -p /root/custom_nodes

# 简化的mount命令
mount -t nfs4 -o rw rh-nfs.runninghub.cn:/data/rh_storage/global/custom_nodes_rel /root/custom_nodes

# 获取当前目录名作为NODE_NAME
NODE_NAME=$(basename "$PWD")

# 创建目标目录
mkdir -p /root/custom_nodes/${NODE_NAME}

# 显示将要执行的rsync命令
echo "准备执行以下rsync命令："
echo "rsync -av --include=\"*/\" --include=\"*.py\" --exclude=\"*\" ./ /root/custom_nodes/${NODE_NAME}/"
echo ""
echo "此命令将同步当前目录及子目录中的所有 .py 文件到 /root/custom_nodes/${NODE_NAME}/"
echo ""
read -p "是否继续执行？(输入 Y 确认): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始同步 Python 文件..."
    rsync -av --include="*/" --include="*.py" --exclude="*" ./ /root/custom_nodes/${NODE_NAME}/
    
    if [ $? -eq 0 ]; then
        echo "Python 文件同步完成！"
    else
        echo "Python 文件同步失败！"
        exit 1
    fi
else
    echo "取消同步操作。"
    exit 1
fi
