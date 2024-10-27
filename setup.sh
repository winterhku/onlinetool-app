#!/bin/bash

mkdir -p ~/.streamlit/

echo "[general]
email = \"rzhaoag@connect.ust.hk\"
" > ~/.streamlit/credentials.toml

echo "[server]
headless = true
enableCORS = false
port = \$PORT
" > ~/.streamlit/config.toml
