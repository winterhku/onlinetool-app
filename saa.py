import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from gurobipy import Model, GRB, quicksum
import numpy as np

# class LBBD:
#     def __init__(self):
#         self.model = None
#         self.data = None
#         self.num_sites = 0
#         self.x = None
#         self.y_fast = None
#         self.y_slow = None
#         self.latitude = []
#         self.longitude = []
#         self.location = []
#         self.max_pumps_per_site = []
#         self.fast_charging_cost = []
#         self.fast_charging_speed = []
#         self.slow_charging_cost = []
#         self.slow_charging_speed = []
#         self.fixed_cost = []
#
#     def set_data(self, data):
#         self.data = data
#         self.num_sites = len(data)
#         self.latitude = data['Latitude'].tolist()
#         self.longitude = data['Longitude'].tolist()
#         self.location = data['Location'].tolist()
#         self.max_pumps_per_site = data['Max_Pumps'].tolist()
#         self.fast_charging_cost = data['Fast_Charging_Cost'].tolist()
#         self.fast_charging_speed = data['Fast_Charging_Speed'].tolist()
#         self.slow_charging_cost = data['Slow_Charging_Cost'].tolist()
#         self.slow_charging_speed = data['Slow_Charging_Speed'].tolist()
#         self.fixed_cost = data['Fixed_Cost'].tolist()
#
#     def load_and_prepare_data(self, budget, max_stations, max_pumps, objective):
#         self.model = Model("OptimizationModel")
#         self.x = self.model.addVars(self.num_sites, vtype=GRB.BINARY, name="x")
#         self.y_fast = self.model.addVars(self.num_sites, vtype=GRB.INTEGER, name="y_fast")
#         self.y_slow = self.model.addVars(self.num_sites, vtype=GRB.INTEGER, name="y_slow")
#
#         for i in range(self.num_sites):
#             self.model.addConstr(self.y_fast[i] + self.y_slow[i] <= self.max_pumps_per_site[i] * self.x[i], name=f"max_pumps_site_{i}")
#         total_pumps = self.y_fast.sum() + self.y_slow.sum()
#         self.model.addConstr(total_pumps <= max_pumps, name="max_total_pumps")
#         self.model.addConstr(self.x.sum() <= max_stations, name="max_stations")
#         total_cost = quicksum(self.x[i] * self.fixed_cost[i] + self.y_fast[i] * self.fast_charging_cost[i] + self.y_slow[i] * self.slow_charging_cost[i] for i in range(self.num_sites))
#         self.model.addConstr(total_cost <= budget, name="budget_constraint")
#
#         if objective == 'Minimize Cost':
#             self.model.setObjective(total_cost, GRB.MINIMIZE)
#         elif objective == 'Maximize Demand Coverage':
#             demand_covered = quicksum(self.y_fast[i] * self.fast_charging_speed[i] + self.y_slow[i] * self.slow_charging_speed[i] for i in range(self.num_sites))
#             self.model.setObjective(demand_covered, GRB.MAXIMIZE)
#         elif objective == 'Simultaneous Optimization':
#             demand_covered = quicksum(self.y_fast[i] * self.fast_charging_speed[i] + self.y_slow[i] * self.slow_charging_speed[i] for i in range(self.num_sites))
#             cost_weight = 1
#             demand_weight = 1
#             self.model.setObjective(demand_weight * demand_covered - cost_weight * total_cost, GRB.MAXIMIZE)
#         self.model.update()
#
#     def optimize(self, time_limit):
#         if not self.model:
#             raise Exception("Model not prepared. Call prepare_model first.")
#         self.model.setParam('TimeLimit', time_limit)
#         self.model.optimize()
#
#         if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
#             print("Optimization completed.")
#         elif self.model.status == GRB.INFEASIBLE:
#             print("Model is infeasible.")
#             self.model.computeIIS()
#             self.model.write("model.ilp")
#         else:
#             print("Optimization was stopped with status", self.model.status)
#
#         optimal_indices = [i for i in range(self.num_sites) if self.x[i].X > 0.5]
#         return optimal_indices
#
#
# import streamlit as st
# from streamlit_folium import folium_static
# from SAA_charger0926 import LBBD  # 确保已经正确导入LBBD类
#
#
# def main():
#     st.title('Charging Station Optimization and Visualization')
#
#     # 定义文件路径
#     filepaths = {
#         'charging_stations': 'C:\\Users\\myhom\\Downloads\\HKU 充电桩优化选址\\输出\\充电桩2km数据统计表v2.0.xlsx',
#         'parking_lots': 'C:\\Users\\myhom\\Downloads\\HKU 充电桩优化选址\\停车场\\hkparking_sample.xlsx',
#         'taxi_stands': 'C:\\Users\\myhom\\Downloads\\HKU 充电桩优化选址\\输出\\的士站2km数据统计表.xlsx',
#         'petrol_stations': 'C:\\Users\\myhom\\Downloads\\HKU 充电桩优化选址\\输出\\加油站2km数据统计表 v2.0.xlsx'
#     }
#
#     st.sidebar.title('Select Candidate Site Types')
#     site_selection = {site: st.sidebar.checkbox(site.replace('_', ' ').title(), value=True) for site in filepaths}
#     selected_filepaths = [path for site, path in filepaths.items() if site_selection[site]]
#
#     # 用户输入的参数
#     budget = st.sidebar.number_input('Enter Budget:', min_value=0, value=1000000)
#     max_stations = st.sidebar.number_input('Max Number of Charging Stations:', min_value=1, value=10)
#     objective_type = st.sidebar.radio('Optimization Objective:', ['Minimize Cost', 'Maximize Demand', 'Balanced'],
#                                       index=2)
#     objective_mapping = {'Minimize Cost': 0, 'Maximize Demand': 1, 'Balanced': 2}
#     objective_mapping = {'Minimize Cost': 0, 'Maximize Demand': 1, 'Balanced': 2}
#     objective_type_value = objective_mapping[objective_type]
#
#     time_limit = st.sidebar.number_input('Optimization Time Limit (seconds):', min_value=30, max_value=300, value=60)
#
#     if st.sidebar.button('Optimize Locations'):
#         if not selected_filepaths:
#             st.error("No candidate sites selected. Please select at least one type of site.")
#             return
#         LBBD_solver = LBBD()
#         # LBBD_solver.set_instance()
#         demand_file_path = 'C:/Users/myhom/Downloads/A_update.csv'
#         parameter_file_path = 'C:/Users/myhom/Downloads/B_update.csv'
#         print(budget)
#         print(max_stations )
#         print(objective_type)
#         print(objective_mapping)
#         print(time_limit)
#         LBBD_solver.read_instance(demand_file_path, parameter_file_path, budget, max_stations,  objective_type_value , time_limit)
#         # input_budget,input_maxstation,input_objtype,input_timelimit,
#         # where input_objtype is {0,1,2} represent
#         # 0: min total cost when area demand is meet but demand of each p is ignored;
#         # 1: max cover demand of each p and rea demand is meet, then to min cost
#         # 2: min total cost when area demand is meet but demand of each p is moderate consideration;
#         totalcost, fixcost, operationscost, uncoverddemand, finalresult = LBBD_solver.build_MIP_model()
#         print("totalcost:", totalcost, "fixcost:", fixcost, "operationscost", operationscost, "uncoverdemand:",
#               uncoverddemand)
#         print("location and size:", finalresult)
#
#
#
#         st.write("Total Cost:", totalcost)
#         st.write("Fixed Cost:", fixcost)
#         st.write("Operations Cost:", operationscost)
#         st.write("Uncovered Demand:", uncoverddemand)
#         st.write("Location and Size:", finalresult)
#
#         # 显示结果的地图，假设LBBD_solver有一个创建地图的方法
#         map_ = LBBD_solver.create_map(finalresult)
#         folium_static(map_)
#     else:
#         st.write('Set parameters and press "Optimize Locations".')
#
#
# if __name__ == '__main__':
#     main()
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
    budget = st.sidebar.number_input('Enter Budget:', min_value=0, value=100000000, help="Each fast charger is 20000 hkd,which is 50-150kW;Each slow charger is 5000 hkd,which is 7kW.")
    max_stations = st.sidebar.number_input('Max Number of Charging Stations:', min_value=1, value=10, help="The maximum number of each charging station can be set is varies from 0-24.")
    objective_type = st.sidebar.radio('Optimization Objective:', ['Minimize Cost', 'Maximize Demand', 'Balanced'],
                                      index=2, help="Balanced means minimize the cost and maximize the demand at the same time.")
    objective_mapping = {'Minimize Cost': 0, 'Maximize Demand': 1, 'Balanced': 2}
    objective_type_value = objective_mapping[objective_type]
    time_limit = st.sidebar.number_input('Optimization Time Limit (seconds):', min_value=30, max_value=300, value=60, help="The maximum time you want to wait for the solver.")

    if st.sidebar.button('Optimize Locations'):
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
        totalcost, fixcost, operationscost, uncoverddemand, finalresult = LBBD_solver.build_MIP_model()

        st.write("Total Cost:",format(totalcost,'.2f'),"$")
        st.write("Fixed Cost:", format(fixcost,'.2f'),"$")
        st.write("Operations Cost:", format(operationscost,'.2f'),"$")
        st.write("Uncovered Demand:", format(uncoverddemand,'.2f'),"$")
        # st.write("Location and Size:", finalresult)

        # 调用create_map来生成地图，传入finalresult和B文件数据
        map_ = LBBD_solver.create_map(finalresult, df_b)
        folium_static(map_)
    else:
        st.write('Set parameters and press "Optimize Locations".')


if __name__ == '__main__':
    main()

# import requests
# import streamlit as st
# import pandas as pd
# import folium
# from streamlit_folium import folium_static
#
# def main():
#     st.set_page_config(page_title="在线工具箱@Smart Mobility Lab", page_icon="🗺", layout="wide")
#     st.title('Charging Station Optimization Tool')
#
#     # 获取用户输入的参数
#     budget = st.sidebar.number_input('Enter Budget:', min_value=0, value=100000000)
#     max_stations = st.sidebar.number_input('Max Number of Charging Stations:', min_value=1, value=10)
#     objective_type = st.sidebar.radio('Optimization Objective:', ['Minimize Cost', 'Maximize Demand', 'Balanced'], index=2)
#     objective_mapping = {'Minimize Cost': 0, 'Maximize Demand': 1, 'Balanced': 2}
#     objective_type_value = objective_mapping[objective_type]
#     time_limit = st.sidebar.number_input('Optimization Time Limit (seconds):', min_value=30, max_value=300, value=60)
#
#     # 本地文件路径
#     demand_file_path = 'C:/Users/myhom/Downloads/A_update.csv'
#     parameter_file_path = 'C:/Users/myhom/Downloads/B_update.csv'
#
#     if st.sidebar.button('Optimize Locations'):
#         payload = {
#             'budget': budget,
#             'max_stations': max_stations,
#             'objective_type_value': objective_type_value,
#             'time_limit': time_limit,
#             'demand_file_path': demand_file_path,
#             'parameter_file_path': parameter_file_path
#         }
#
#         # 向API发送POST请求
#         try:
#             response = requests.post("http://localhost:8080/optimize", json=payload)
#             if response.status_code == 200:
#                 result = response.json()
#
#                 # 显示结果
#                 st.write("Total Cost:", result.get('TotalCost'), "$")
#                 st.write("Fixed Cost:", result.get('FixedCost'), "$")
#                 st.write("Operations Cost:", result.get('OperationsCost'), "$")
#                 st.write("Uncovered Demand:", result.get('UncoveredDemand'), "$")
#
#                 # 显示地图
#                 finalresult = result.get('FinalResult')
#                 df_b = pd.read_csv(parameter_file_path)
#                 map_ = create_map(finalresult, df_b)
#                 folium_static(map_)
#             else:
#                 st.write(f"Error in optimization process: {response.status_code}")
#         except Exception as e:
#             st.write(f"Failed to connect to API: {e}")
#
# def create_map(finalresult, df_b):
#     map_ = folium.Map(location=[22.3193, 114.1694], zoom_start=12)
#     for result in finalresult:
#         location = [result['latitude'], result['longitude']]
#         folium.Marker(location).add_to(map_)
#     return map_
#
# if __name__ == '__main__':
#     main()

