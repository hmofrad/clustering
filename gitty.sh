#!/bin/bash
# Committing made easy
# Working with origin remote repository and master branch 
# ./gitty FILE_TO_COMMIT ["COMMIT_COMMENTS"]
# (C) Mohammad H. Mofrad, 2016

# To setup under a Linux box after git clone REPO_ADDRESS
# git config --global user.email EMAIL
# git config --system user.email EMAIL
# git config --local  user.email EMAIL

if [ -z "$1" ] || [ $# -gt 2 ]; then
    echo "Usage: ./gitty.sh FILE [\"COMMENT\"]";
    exit 1;
fi

FILE=$1
COMMENT=$2
# Disable gnome-ssh-askpass
unset SSH_ASKPASS;
# File to commit
git add $FILE;
# Now commit the change
git commit -m "$FILE: $COMMENT";
# Push the commit to master branch
git push origin master;
