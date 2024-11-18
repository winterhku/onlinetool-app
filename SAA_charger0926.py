"""
"""

from pyscipopt import Model
import copy
import time
import numpy as np
import pandas as pd
import random
import folium 

class LBBD(object):
    def __init__(self):
        self.model = None
        self.BendersMP = None
        self.BendersSP = None

        self.x_MP_sol = {}
        self.y_MP_sol = {}
        self.x_MP_previous = {}
        self.y_MP_previous = {}
        self.x_pie_MP_sol = {}
        self.x_pie_MP_previous = {}

        self.global_LB = 0
        self.global_UB = 5000000

        self.Gap = 1000

        self.iter_cnt = 0

        self.LB = 400000

        self.start_time = time.time()
        self.end_time = time.time()
        self.CPU_time = 0

    def set_instance(self):
        self.T = 24  # Period
        self.A = 17  # Number of area:屯门	元朗	荃湾	深水埗	油尖旺	九龙城	北区	大埔	中西区	沙田	黄大仙	东区	观塘	离岛	南区	西贡	葵青
        self.P = 100  # Number of candidate stations
        self.scenario = 1  # FIXME: 没有随机性，设置成1
        # random generate station id
        self.station_points = list(range(self.P))
        np.random.seed(42)
        self.regions = [[] for _ in range(
            self.A)]  # record the ownership, This is randomly generated, and you need to replace it with read data
        for point in self.station_points:
            region_index = np.random.randint(0, self.A)
            self.regions[region_index].append(point)

        self.F = (np.random.randint(50000, 100000, self.P) / 365).astype(int)  # rebuilding cost in p station
        self.C1 = (np.random.randint(4000, 8000, self.P) / 30).astype(int)  # fix recent cost in p station
        self.C2 = (np.array([30000, 100000]) / 365).astype(int)  # buy cost
        # self.C2 = (np.array([30000, 50000, 100000]) / 365).astype(int)  # bug cost
        self.N = [2, 12]  # service capacity per unit time of a super charger  # FIXME: 快慢充，可以设置成三个但需要注释掉有效不等式
        # self.N = [1, 2, 12]  # service capacity per unit time of a super charger
        self.C = np.add.outer(self.C1, self.C2)  # [p,k]
        self.K = len(self.C2)
        self.D = 0.5 * np.random.randint(600, 800,
                                         (self.A, self.T, self.K, self.scenario))  # demand in area a in time t
        self.D_p = 0.5 * np.random.randint(40, 80,
                                           (self.P, self.T, self.K, self.scenario))  # demand in candidation p in time t
        self.R = np.round(np.random.random(self.P),
                          decimals=1)  # the distance rate from station p to the center of its area
        self.U = np.random.randint(50, 100, self.P)  # The maximum number of charging piles allowed in station p
        self.gamma = np.round(0.1 * np.random.random(
            (self.P, self.T, self.K)),
                              decimals=1)  # Minimum coverage demand ratio in t period in station p # FIXME: 设置的太大会导致无解
        self.B = 0.5 * np.random.randint(20, 60, (
        self.P, self.K))  # unit time charging capacity of existing charging piles in station p
        self.B_a = 0.5 * np.random.randint(400, 600, (
        self.A, self.K))  # unit time charging capacity of existing charging piles in area A
        self.alpha = np.round(0.05 * np.random.random((self.P, self.T, self.K)),
                              decimals=1)  # increase rate due to new chargers]\
        self.W = np.random.randint(400, 800, (self.P, self.T))
        # self.belta = np.round(np.random.uniform(0.6, 0.9, self.A),2)
        # self.phi = np.round(np.random.uniform(0.1, 0.2, (self.P, self.T, self.K)),decimals=1)
        self.L = 10.24  # capacity
        self.l = np.random.randint(4, 5, (self.T))
        self.CC = (np.random.randint(36000, 42000, self.P) / 365).astype(int)

    def read_instance(self, demand_file_path, parameter_file_path, input_budget, input_maxstation, input_objtype,
                      input_timelimit):
        # print("ffffffffffffffff===============ffffffff")
        demand_df = pd.read_csv(demand_file_path).set_index(["id", "Hour"])

        parameter_df = pd.read_csv(parameter_file_path)
        parameter_df['Average_Property_Rent'].fillna(parameter_df['Average_Property_Rent'].mean(),
                                                     inplace=True)  # 月租金空缺处填均值
        self.T = 24  # Period
        self.A = 18  # Number of area:(屯门)	(元朗)	(荃湾)	(深水埗)	(油尖旺)	(九龙城)	(北区)	(大埔)	(中西区)	(沙田)	(黄大仙)	(东区)	(观塘)	(离岛)	(南区)	(西贡)	(葵青) (湾仔)
        district = ["KOWLOON CITY", "YAU TSIM MONG", "SHAM SHUI PO", "KWUN TONG", "WONG TAI SIN", "EASTERN",
                    "CENTRAL AND WESTERN", "SOUTHERN", "SAI KUNG", "SHA TIN", "TUEN MUN", "TSUEN WAN", "YUEN LONG",
                    "TAI PO", "NORTH", "KWAI TSING", "ISLAND", "WAN CHAI"]
        district_to_id_dict = {d: i for (i, d) in enumerate(district)}
        low_ratio = [0.421568627, 0.677570093, 0.459915612, 0.415948276, 0.50390625, 0.710526316, 0.769911504,
                     0.794871795, 0.539772727, 0.558201058, 0.571428571, 0.418848168, 0.795646917, 0.534759358,
                     0.227272727, 0.458715596, 0.413333333,
                     0.56561086]  # NOTE: 慢充比例，源自分区域发展目标.xlsx，中速按一半分别折算进快慢，分区与上面的注释同顺序
        self.MAX = input_maxstation
        self.P = len(parameter_df)  # Number of candidate stations  # NOTE: 候选充电站数量
        self.scenario = 1  # NOTE: 没有随机性，设置成1
        # random generate station id
        self.station_points = list(range(self.P))
        np.random.seed(42)
        self.regions = [[] for _ in range(
            self.A)]  # record the ownership, This is randomly generated, and you need to replace it with read data
        for index, row in parameter_df.iterrows():
            district = row['District']
            district_id = district_to_id_dict[district]
            self.regions[district_id].append(index)  # NOTE: 这里regions[index]是所属区域为index的充电站的id
        # for point in self.station_points:
        #     region_index = np.random.randint(0, self.A)
        #     self.regions[region_index].append(point)

        # self.F = (np.random.randint(50000, 100000, self.P) / 365).astype(int)  # rebuilding cost in p station
        self.F = parameter_df["Average_Property_Rent"].values / 30  # NOTE: 建设成本，用月租金摊到天上
        self.C1 = (np.random.randint(4000, 8000, self.P) / 30).astype(int)  # fix recent cost in p station # TODO:
        self.C2 = (np.array([5000, 20000]) / 365).astype(int)  # buy cost  # NOTE: 买充电桩的价格 先慢冲后快充
        self.N = [1, 6]  # service capacity per unit time of a super charger  # NOTE: 慢快充每小时能服务的数量
        self.C = np.add.outer(self.C1, self.C2)  # [p,k]
        self.K = len(self.C2)
        self.finalresult = np.zeros((len(parameter_df), self.K))

        self.D_p = np.zeros(
            (self.P, self.T, self.K, self.scenario))  # demand in candidation p in time t  # NOTE: 每个P点慢冲/快冲需求
        for index, row in demand_df.iterrows():
            p, t = index
            demand = row["Total Demand Count"]
            p_belong_to_district = district_to_id_dict[parameter_df.loc[p, "District"]]
            self.D_p[p, t, 0, 0] = demand * low_ratio[p_belong_to_district]  # NOTE: 慢充需求
            self.D_p[p, t, 1, 0] = demand * (1 - low_ratio[p_belong_to_district])  # NOTE: 快充需求
        self.D_p = self.D_p / 24 / 2.6
        self.D = np.zeros((self.A, self.T, self.K, self.scenario))  # demand in area a in time t  # NOTE: 每个A区域内慢冲/快冲需求
        for region in range(self.A):
            self.D[region] = np.sum(self.D_p[self.regions[region]], axis=0)
        self.R = np.round(np.random.random(self.P),
                          decimals=1)  # the distance rate from station p to the center of its area

        # self.U = np.random.randint(50, 100, self.P)  # The maximum number of charging piles allowed in station p
        self.U = 2000000 * (parameter_df["最大可设置充电桩数量"].values + 1)  # NOTE: 充电桩数量上限
        self.B = np.zeros((self.P,
                           self.K))  # unit time charging capacity of existing charging piles in station p  # NOTE: 每个P点慢冲/快冲每小时能服务的车的数量
        for index, row in parameter_df.iterrows():
            p_belong_to_district = district_to_id_dict[row["District"]]
            ratio = low_ratio[p_belong_to_district]
            self.B[index] = row["Total_Capacity"] * np.array([ratio, 6 * (1 - ratio)])  # NOTE: 每个P点慢冲/快冲每小时能服务的车的数量

        self.B_a = np.zeros((self.A,
                             self.K))  # unit time charging capacity of existing charging piles in area A  # NOTE: 每个A区域内慢冲/快冲每小时能服务的车的数量
        for region in range(self.A):
            self.B_a[region] = np.sum(self.B[self.regions[region]], axis=0)

        self.gamma = np.round(0.7 * np.random.random((self.P, self.T, self.K)),
                              decimals=3)  # Minimum coverage demand ratio in t period in station p #设置的太大会导致无解 NOTE: 最小覆盖率 超参先不管
        self.alpha = np.round(0.7 * np.random.random((self.P, self.T, self.K)),
                              decimals=3)  # increase rate due to new chargers NOTE: 增长率 超参先不管
        self.W = np.random.randint(1e8, 1.1e8, (self.P, self.T))  # NOTE: 电网能力，暂时放到无限大
        self.L = 10.24  # capacity # NOTE: 光伏储能容量10.24度
        self.l = np.random.randint(2, 3, (self.T))  # NOTE: 光伏充电量4度/小时
        self.CC = (np.random.randint(36000, 42000, self.P) / 365).astype(int)  # NOTE: 光伏充电桩价格3w6-4w2
        self.budget = input_budget
        self.maxnumber = input_maxstation
        self.D_one = 57.6  # 一个电动车最大电容量72千瓦时，假设单次充电为80%，
        self.timelimit = input_timelimit
        self.objtype = input_objtype

    def build_MIP_model(self):
        """ 设置算例"""
        # self.set_instance()
        self.x = {}
        # self.x_pie = {}
        self.y = {}
        self.z = {}
        self.w = {}
        self.w_k = {}
        self.q = {}
        self.q2 = {}
        # self.e = {}
        # self.s = {}
        self.model = Model("milp")
        for p in range(self.P):
            self.y[p] = self.model.addVar(vtype="BINARY", name="y_" + str(p))

            # self.x_pie[p] = self.model.addVar(vtype=GRB.INTEGER, name="x_pie_" + str(p))
            for k in range(self.K):
                self.x[p, k] = self.model.addVar(lb=0,vtype="INTEGER", name="x_" + str(p) + "_" + str(k))
            for t in range(self.T):
                for sw in range(self.scenario):
                    # self.s[p, t, sw] = self.model.addVar(lb=0, ub=np.inf, vtype=GRB.CONTINUOUS, name="s_" + str(p) + "_" + str(t))
                    # self.e[p, t, sw] = self.model.addVar(lb=0, ub=np.inf, vtype=GRB.CONTINUOUS, name="e_" + str(p) + "_" + str(t))
                    self.z[p, t, sw] = self.model.addVar(lb=0,vtype="CONTINUOUS", name="z_" + str(p) + "_" + str(t))
                    self.w[p, t, sw] = self.model.addVar(lb=0,vtype="CONTINUOUS", name="w_" + str(p) + "_" + str(t))
                    for k in range(self.K):
                        for k2 in range(self.K):
                            if k <= k2:
                                self.w_k[p, t, k, k2, sw] = self.model.addVar(lb=0,vtype="CONTINUOUS",
                                                                              name="w_k_" + str(p) + "_" + str(
                                                                                  t) + "_" + str(k) + "_" + str(k2))

        zero_var = self.model.addVar("zero_var", lb=0, ub=0)  # 定义一个固定值为0的辅助变量
# 添加最大值约束：q = max(q2, 0)

        M = 100000000  # 可以根据实际情况调整 M 的值

        # 添加变量和约束
        for t in range(self.T):
            for p in range(self.P):
                for k2 in range(self.K):
                    for sw in range(self.scenario):
                        # 定义连续变量 q 和 q2
                        self.q[p, t, k2, sw] = self.model.addVar(lb=-np.inf, ub=np.inf, vtype="CONTINUOUS",
                                                                 name=f"q_{p}_{t}_{k2}_{sw}")
                        self.q2[p, t, k2, sw] = self.model.addVar(lb=-np.inf, ub=np.inf, vtype="CONTINUOUS",
                                                                  name=f"q2_{p}_{t}_{k2}_{sw}")
                        # 定义二进制变量 delta
                        delta = self.model.addVar(vtype="BINARY", name=f"delta_{p}_{t}_{k2}_{sw}")

                        # 添加线性化的最大值约束
                        # q >= q2
                        self.model.addCons(self.q[p, t, k2, sw] >= self.q2[p, t, k2, sw])
                        # q >= 0
                        self.model.addCons(self.q[p, t, k2, sw] >= zero_var)
                        # q <= q2 + M * delta
                        self.model.addCons(self.q[p, t, k2, sw] <= self.q2[p, t, k2, sw] + M * delta)
                        # q <= M * (1 - delta)
                        self.model.addCons(self.q[p, t, k2, sw] <= M * (1 - delta))
        # obj = LinExpr(0)
        # obj2 = LinExpr(0)
        # for p in range(self.P):
        #     obj.addTerms(self.F[p], self.y[p])
        #     # obj.addTerms(self.CC[p], self.x_pie[p])
        #     for k in range(self.K):
        #         obj.addTerms(self.C[p, k], self.x[p, k])
        #     for t in range(self.T):
        #         for sw in range(self.scenario):
        #             obj2.addTerms(1, self.z[p, t, sw])

        obj = 0  # 用于存储目标1的项
        obj2 = 0  # 用于存储目标2的项
        obj3 = 0  # 用于存储目标3的项

        # 构造 obj (类似于 obj = LinExpr(0))
        for p in range(self.P):
            obj += self.F[p] * self.y[p]  # 累加固定成本项
            for k in range(self.K):
                obj += self.C[p, k] * self.x[p, k]  # 累加变量成本项
            for t in range(self.T):
                for sw in range(self.scenario):
                    obj2 += self.z[p, t, sw]  # 累加 z 项

        # 构造 obj3 (类似于 obj3 = LinExpr(0))
        for t in range(self.T):
            for p in range(self.P):
                for k2 in range(self.K):
                    for sw in range(self.scenario):
                        obj3 += self.q[p, t, k2, sw]  # 累加 q 项

        if self.objtype == 0:  # min cost but big area is meet
            self.model.setObjective(obj + obj2, sense="minimize")  ##obj1
        elif self.objtype == 1:  # max area demand but don't care cost:
            self.model.setObjective(obj + obj2 + 10000 * obj3, sense="minimize")  ##obj2
        elif self.objtype == 2:  # balance
            self.model.setObjective(obj + obj2 + 50 * obj3, sense="minimize")  ##obj3 where 50是一个惩罚因子，越大越说明cover demand越重要

        lhs = sum(self.y[p] for p in range(self.P))
        self.model.addCons(lhs <= self.maxnumber)

        # Constraint 1: obj + (1 / scenario) * obj2 <= budget
        obj = sum(self.F[p] * self.y[p] for p in range(self.P))  # Replace with actual objective expression if needed
        obj2 = sum(self.z[p, t, sw] for p in range(self.P) for t in range(self.T) for sw in range(self.scenario))
        self.model.addCons(obj + (1 / self.scenario) * obj2 <= self.budget)

        # Effective inequalities (translated from "有效不等式")
        for t in range(self.T):
            for sw in range(self.scenario):
                for a in range(self.A):
                    for k2 in range(self.K):
                        lhs = 0
                        for p in self.regions[a]:
                            for k in range(self.K):
                                if k <= k2:
                                    lhs += self.N[k] * (1 - self.alpha[p, t, k]) * self.x[p, k]
                        self.model.addCons(lhs >= self.D[a, t, k2, sw] - self.B_a[a, k2], name="effective_constraint_0")

        for t in range(self.T):
            for sw in range(self.scenario):
                for a in range(self.A):
                    lhs = sum(self.N[k] * (1 - self.alpha[p, t, k]) * self.x[p, k] for p in self.regions[a] for k in
                              range(self.K))
                    rhs = sum(self.D[a, t, k, sw] - self.B_a[a, k] for k in range(self.K))
                    self.model.addCons(lhs >= rhs, name=f"effective_constraint_1_{a}")

        # Area demand constraint
        for t in range(self.T):
            for sw in range(self.scenario):
                for a in range(self.A):
                    for k2 in range(self.K):
                        lhs = sum((1 - self.alpha[p, t, k]) * self.w_k[p, t, k, k2, sw] for p in self.regions[a] for k in
                                  range(self.K) if k <= k2)
                        self.model.addCons(lhs + self.B_a[a, k2] >= self.D[a, t, k2, sw], name="area_demand_constraint")

        # Modified constraint (您提到 "constraint 修改版")
        for t in range(self.T):
            for sw in range(self.scenario):
                for p in range(self.P):
                    for k2 in range(self.K):
                        lhs = sum(self.w_k[p, t, k, k2, sw] for k in range(self.K) if k <= k2)
                        self.model.addCons(self.q2[p, t, k2, sw] >= -lhs - self.B[p, k2] + self.gamma[p, t, k2] * (
                                    self.D_p[p, t, k2, sw] + self.alpha[p, t, k2] * lhs))

        # Constraints 3 and 4
        for p in range(self.P):
            for sw in range(self.scenario):
                lhs = sum(self.x[p, k] for k in range(self.K))
                self.model.addCons(lhs <= self.U[p] * self.y[p], name="constraint_3")
                self.model.addCons(self.y[p] <= lhs, name="constraint_4")

        for t in range(self.T):
            for sw in range(self.scenario):
                for p in range(self.P):
                    # SCIP does not have addGenConstrPWL, so we need to create piecewise linear constraints manually.
                    # This is a piecewise approximation: (0,0), (1000,100), (2000,300), (3000,600)
                    # Define auxiliary variables and add the required piecewise constraints
                    # Add constraints to enforce piecewise linear behavior
                    # Here we approximate:
                    # if 0 <= self.w[p, t, sw] <= 1000, then self.z[p, t, sw] = 0.1 * self.w[p, t, sw]
                    self.model.addCons(self.z[p, t, sw] == 1.2 * self.w[p, t, sw], "piecewise_1")
                    # Other regions can be similarly added with additional constraints.

        # # # 手动线性化f_{pt}\left( w_{pt} \right) =w_{pt}\lambda _1+\left( 5w_{pt}\lambda _2+100\lambda _2 \right) +\left( 10w_{pt}\lambda _3+600\lambda _3 \right)
        # self.wz_f = {}
        # self.wz_ff = {}
        # for p in range(self.P):
        #     for t in range(self.T):
        #         for sw in range(self.scenario):
        #             for this in range(3):
        #                 self.wz_f[p, t, sw, this] = self.model.addVar(vtype="BINARY",
        #                                                               name="wzf_" + str(p) + str(t) + str(this))
        #                 self.wz_ff[p, t, sw, this] = self.model.addVar(lb=0, ub=1000000000, vtype="CONTINUOUS",
        #                                                                name="wzff_" + str(p) + str(t) + str(this))
        #             self.model.addCons(
        #                 self.z[p, t, sw] == 1*self.wz_f[p, t, sw, 0] * (self.w[p, t, sw]) + self.wz_f[p, t, sw, 1] * (
        #                             2 * self.w[p, t,sw] + 1000) + self.wz_f[p, t,sw, 2] * (3 * self.w[p, t,sw] + 3000))
        #             for this in range(3):
        #                 self.model.addCons(self.wz_ff[p, t, sw,this] <= 1000000000*self.wz_f[p,t,sw,this])
        #                 self.model.addCons(self.wz_ff[p, t, sw,this] <=self.w[p,t, sw])
        #                 self.model.addCons(self.wz_ff[p, t, sw,this] >= self.w[p, t, sw]-1000000000*(1-self.wz_f[p,t,sw,this]))
        #
        #             self.model.addCons(self.wz_f[p,t,sw,0]+self.wz_f[p,t,sw,1]+self.wz_f[p,t,sw,2]==1)
        #             self.model.addCons(self.w[p, t,sw]>=0)
        #             self.model.addCons(self.w[p, t,sw] <=100000000*(1-self.wz_f[p,t,sw,0])+1000)
        #             self.model.addCons(self.w[p, t,sw] >= self.wz_f[p, t, sw,1]* 100)
        #             self.model.addCons(self.w[p, t,sw] <= 100000000*(1-self.wz_f[p,t,sw,1])+3000)
        #             self.model.addCons(self.w[p, t,sw] >= self.wz_f[p, t, sw,2] * 200)
        #             self.model.addCons(self.w[p, t,sw] <= 10000000 * (1 - self.wz_f[p, t,sw, 2]) + 300000000)
        for p in range(self.P):
            for sw in range(self.scenario):
                for t in range(self.T):
                    for k in range(self.K):
                        llhs =0
                        for k2 in range(self.K):
                            if k <= k2:
                                llhs+= self.w_k[p, t, k, k2, sw]
                        self.model.addCons(llhs <= self.N[k] * self.x[p, k], name="cons7_")  # constraint 7

        for t in range(self.T):
            for sw in range(self.scenario):
                for p in range(self.P):
                    lhs = 0
                    for k in range(self.K):
                        for k2 in range(self.K):
                            if k <= k2:
                                lhs += self.w_k[p, t, k, k2, sw]
                    self.model.addCons(self.w[p, t, sw] == self.D_one * lhs)

        # model.computeIIS()

        # self.model.write("chargermodel0731.lp")
        self.model.setParam("limits/gap", 0.0001)  # 设置 MIPGap
        self.model.setParam("limits/time", self.timelimit)  # 设置 TimeLimit
        self.model.optimize()

        if self.model.getStatus() == "infeasible":
            # SCIP 不需要 computeIIS 和 write，因为您提到不需要 IIS 分析
            self.finalresult = "No solution"
            # 返回 0, 0, 0, 0 和结果信息
            return 0, 0, 0, 0, self.finalresult
        else:
            obj_value = sum(self.F[p] * self.model.getVal(self.y[p]) for p in range(self.P))  # 计算 obj
            obj2_value = sum(self.model.getVal(self.z[p, t, sw]) for p in range(self.P) for t in range(self.T) for sw in
                             range(self.scenario))  # 计算 obj2
            obj3_value = sum(
                self.model.getVal(self.q[p, t, k2, sw]) for t in range(self.T) for p in range(self.P) for k2 in
                range(self.K) for sw in range(self.scenario))  # 计算 obj3
            for p in range(self.P):
                for k in range(self.K):
                    self.finalresult[p, k] =self.model.getVal(self.x[p, k])

            # 返回计算结果
        return obj_value + obj2_value, obj_value, obj2_value, obj3_value, self.finalresult
            # 获取解并存储在 self.finalresult 中

    import folium

    def create_map(self, locations_and_sizes, df_b):
        # 创建基础地图（以香港为例）
        base_map = folium.Map(location=[22.3193, 114.1694], zoom_start=12)

        # 遍历每个被选中的地点及其规模
        for index, (slow_charge_num, fast_charge_num) in enumerate(locations_and_sizes):
            # 仅当慢充或快充数目不为0时才处理
            if slow_charge_num > 0 or fast_charge_num > 0:
                # 使用索引在B文件中查找地点的经纬度
                location_data = df_b[df_b['id'] == index]
                if not location_data.empty:
                    lat = location_data.iloc[0]['Latitude']
                    lng = location_data.iloc[0]['Longitude']

                    # 根据规模大小设置标记的颜色或大小
                    total_size = slow_charge_num + fast_charge_num
                    if total_size > 5:
                        color = 'red'
                        radius = 10
                    elif total_size > 1:
                        color = 'blue'
                        radius = 7
                    else:
                        color = 'green'
                        radius = 5

                    # 在地图上添加标记，使用CircleMarker来标识地点
                    folium.CircleMarker(
                        location=(lat, lng),
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_opacity=0.6,
                        popup=f'Slow: {format(slow_charge_num,'.0f')}, Fast: {format(fast_charge_num,'.0f')}'
                    ).add_to(base_map)

        return base_map


if __name__ == '__main__':
    LBBD_solver = LBBD()
    demand_file_path = './A_update.csv'
    parameter_file_path = './B_update.csv'
    # LBBD_solver.set_instance()
    LBBD_solver.read_instance(demand_file_path, parameter_file_path, 100000000, 100, 2,
                              60)  # input_budget,input_maxstation,input_objtype,input_timelimit,
    # where input_objtype is {0,1,2} represent
    # 0: min total cost when area demand is meet but demand of each p is ignored;
    # 1: max cover demand of each p and rea demand is meet, then to min cost
    # 2: min total cost when area demand is meet but demand of each p is moderate consideration;
    totalcost, fixcost, opeationscost, uncoverddemand, finalresult = LBBD_solver.build_MIP_model()
    print("____________________")
    print("totalcost:", totalcost, "fixcost:", fixcost, "operationscost", opeationscost, "uncoverdemand:",
          uncoverddemand)
    print("location and size:", finalresult)
