mkdir -p ~/.streamlit/
echo“\ 
[general] \n\ 
email = \” rzhaoag@connect.ust.hk \“\n\ 
” > ~/.streamlit/credentials.toml
echo "\ 
[服务器]\n\ 
headless = true\n\ 
enableCORS=false\n\ 
port = $PORT\n\ 
" > ~/.streamlit/config.toml