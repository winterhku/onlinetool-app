mkdir -p ~/.streamlit/

echo "\
[general]
email = \"rzhaoag@connect.ust.hk\"
" > ~/.streamlit/credentials.toml

echo "\
[server]
headless = true
port = $PORT
enableCORS = true
enableXsrfProtection = false
" > ~/.streamlit/config.toml
