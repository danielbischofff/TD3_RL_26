#!/bin/bash

export COMPRL_SERVER_URL=comprl.cs.uni-tuebingen.de
export COMPRL_SERVER_PORT=65335
export COMPRL_ACCESS_TOKEN=f68284a1-f8b1-4a1d-a716-f342e9b97a06

python3 ./run_client.py --args --agent=td3