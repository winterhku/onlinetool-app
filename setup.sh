mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"rzhaoag@connect.ust.hk\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
