import pandas as pd
import folium
import streamlit as st
from streamlit_folium import folium_static
from SAA_charger0926 import LBBD  # 确保已经正确导入LBBD类



def main():
    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "000000-Black",
            "height": 720,
            "width": 480,
            "scale": 6,
        }
    }

    icons = {
        "assistant": "https://raw.githubusercontent.com/sahirmaharaj/exifa/2f685de7dffb583f2b2a89cb8ee8bc27bf5b1a40/img/assistant-done.svg",
        "user": "https://raw.githubusercontent.com/sahirmaharaj/exifa/2f685de7dffb583f2b2a89cb8ee8bc27bf5b1a40/img/user-done.svg",
    }

    particles_js = """<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Particles.js</title>
      <style>
      #particles-js {
        position: fixed;
        width: 100vw;
        height: 100vh;
        top: 0;
        left: 0;
        z-index: -1; /* Send the animation to the back */
      }
      .content {
        position: relative;
        z-index: 1;
        color: white;
      }

    </style>
    </head>
    <body>
      <div id="particles-js"></div>
      <div class="content">
        <!-- Placeholder for Streamlit content -->
      </div>
      <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
      <script>
        particlesJS("particles-js", {
          "particles": {
            "number": {
              "value": 300,
              "density": {
                "enable": true,
                "value_area": 800
              }
            },
            "color": {
              "value": "#ffffff"
            },
            "shape": {
              "type": "circle",
              "stroke": {
                "width": 0,
                "color": "#000000"
              },
              "polygon": {
                "nb_sides": 5
              },
              # "image": {
              #   "src": "img/github.svg",
              #   "width": 100,
              #   "height": 100
              # }
            },
            "opacity": {
              "value": 0.5,
              "random": false,
              "anim": {
                "enable": false,
                "speed": 1,
                "opacity_min": 0.2,
                "sync": false
              }
            },
            "size": {
              "value": 2,
              "random": true,
              "anim": {
                "enable": false,
                "speed": 40,
                "size_min": 0.1,
                "sync": false
              }
            },
            "line_linked": {
              "enable": true,
              "distance": 100,
              "color": "#ffffff",
              "opacity": 0.22,
              "width": 1
            },
            "move": {
              "enable": true,
              "speed": 0.2,
              "direction": "none",
              "random": false,
              "straight": false,
              "out_mode": "out",
              "bounce": true,
              "attract": {
                "enable": false,
                "rotateX": 600,
                "rotateY": 1200
              }
            }
          },
          "interactivity": {
            "detect_on": "canvas",
            "events": {
              "onhover": {
                "enable": true,
                "mode": "grab"
              },
              "onclick": {
                "enable": true,
                "mode": "repulse"
              },
              "resize": true
            },
            "modes": {
              "grab": {
                "distance": 100,
                "line_linked": {
                  "opacity": 1
                }
              },
              "bubble": {
                "distance": 400,
                "size": 2,
                "duration": 2,
                "opacity": 0.5,
                "speed": 1
              },
              "repulse": {
                "distance": 200,
                "duration": 0.4
              },
              "push": {
                "particles_nb": 2
              },
              "remove": {
                "particles_nb": 3
              }
            }
          },
          "retina_detect": true
        });
      </script>
    </body>
    </html>
    """
    st.set_page_config(page_title="在线工具箱@Smart Mobility Lab", page_icon="🗺", layout="wide")
    st.title('Charging Station Optimization Tool', help="Can help you optimize the optimal deployment plan for charging facility")

    # 用户输入的参数
    budget = st.sidebar.number_input('Maximum Acceptable Budget Value:', min_value=0, value=100000000, help="Each fast charger is 20000 hkd,which is 50-150kW;Each slow charger is 5000 hkd,which is 7kW.")
    max_stations = st.sidebar.number_input('Maximum Number of Planned Charging Stations:', min_value=1, value=10, help="The maximum number of each charging station can be set is varies from 0-24.")
    objective_type = st.sidebar.radio('Optimization Objective:', ['Minimize Cost', 'Maximize Demand', 'Balanced'],
                                      index=2, help="Balanced means minimize the cost and maximize the demand at the same time.")
    objective_mapping = {'Minimize Cost': 0, 'Maximize Demand': 1, 'Balanced': 2}
    objective_type_value = objective_mapping[objective_type]
    time_limit = st.sidebar.number_input('Optimization Time Limit (seconds):', min_value=30, max_value=300, value=60, help="The maximum time you want to wait for the solver.")

    if st.sidebar.button('Start optimize'):
        LBBD_solver = LBBD()

        # 读取数据文件
        demand_file_path = './A_update.csv'
        parameter_file_path = './B_update.csv'

        # 读取A文件
        df_a = pd.read_csv(demand_file_path)
        # 读取B文件
        df_b = pd.read_csv(parameter_file_path)

        LBBD_solver.read_instance(demand_file_path, parameter_file_path, budget, max_stations, objective_type_value,
                                  time_limit)

        # 运行优化模型并获取结果
        totalcost, fixcost, operationscost, uncoverddemand,countstation, finalresult = LBBD_solver.build_MIP_model()

        st.write("Total Cost:",format(totalcost,'.2f'),"$")
        st.write("Fixed Cost:", format(fixcost,'.2f'),"$")
        st.write("Operations Cost:", format(operationscost,'.2f'),"$")
        st.write("Uncovered Demand:", format(uncoverddemand,'.0f'))
        st.write("Station Number is:",format(countstation,'.0f'))
        # st.write("Location and Size:", finalresult)

        # 调用create_map来生成地图，传入finalresult和B文件数据
        map_ = LBBD_solver.create_map(finalresult, df_b)
        folium_static(map_)
    else:
        st.write('Set parameters and press "Start Optimize".')


if __name__ == '__main__':
    main()
