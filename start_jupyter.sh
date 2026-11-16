#!/usr/bin/env bash
# JupyterLab for BTC's timing check (predict_notebook.ipynb, "Run all"):
#   docker run -it --gpus '"device=0"' -p 9777:9777 -v <data>:/data:ro -v <result>:/result IMAGE /bin/bash /code/start_jupyter.sh
# then open http://localhost:9777 and enter zac2026 as password/token.
cd /code
exec jupyter lab --port 9777 --ip 0.0.0.0 --ServerApp.root_dir=/code \
  --ServerApp.token='zac2026' --IdentityProvider.token='zac2026' --NotebookApp.token='zac2026' \
  --PasswordIdentityProvider.hashed_password="$(python3 -c 'from jupyter_server.auth import passwd; print(passwd("zac2026"))')" \
  --allow-root --no-browser
