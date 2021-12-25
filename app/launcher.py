import os, confuse

config = confuse.Configuration('RecLauncher')
config.set_file('config-st.yaml')
server_port = config['streamlit']['server_port'].get()
os.system(f"streamlit run app.py --server.port {server_port}")