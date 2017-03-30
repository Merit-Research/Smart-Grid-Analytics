#!/bin/bash
createTunnel() {
  /usr/bin/ssh -i _SSH_ID_RSA_PATH_ -N -R 4444:localhost:22 _HOME_SERVER_CALLING_TO_ &
  if [[ $? -eq 0 ]]; then
    echo Tunnel to jumpbox created successfully
  else
    echo An error occurred creating a tunnel to jumpbox. RC was $?
  fi
}
#/bin/pidof ssh
date
/usr/bin/pgrep -x ssh
if [[ $? -ne 0 ]]; then
  echo Creating new tunnel connection
  createTunnel
fi
