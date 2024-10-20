"""
"""

from gurobipy import *
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
        if 'Average_Property_Rent' not in parameter_df.columns:
            parameter_df['Average_Property_Rent'] = 5000  # Default average property rent if not present
        else:
            parameter_df['Average_Property_Rent'].fillna(parameter_df['Average_Property_Rent'].mean(), inplace=True)

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
            self.y[p] = self.model.addVar(vtype=GRB.BINARY, name="y_" + str(p))

            # self.x_pie[p] = self.model.addVar(vtype=GRB.INTEGER, name="x_pie_" + str(p))
            for k in range(self.K):
                self.x[p, k] = self.model.addVar(vtype=GRB.INTEGER, name="x_" + str(p) + "_" + str(k))
            for t in range(self.T):
                for sw in range(self.scenario):
                    # self.s[p, t, sw] = self.model.addVar(lb=0, ub=np.inf, vtype=GRB.CONTINUOUS, name="s_" + str(p) + "_" + str(t))
                    # self.e[p, t, sw] = self.model.addVar(lb=0, ub=np.inf, vtype=GRB.CONTINUOUS, name="e_" + str(p) + "_" + str(t))
                    self.z[p, t, sw] = self.model.addVar(vtype=GRB.CONTINUOUS, name="z_" + str(p) + "_" + str(t))
                    self.w[p, t, sw] = self.model.addVar(vtype=GRB.CONTINUOUS, name="w_" + str(p) + "_" + str(t))
                    for k in range(self.K):
                        for k2 in range(self.K):
                            if k <= k2:
                                self.w_k[p, t, k, k2, sw] = self.model.addVar(vtype=GRB.CONTINUOUS,
                                                                              name="w_k_" + str(p) + "_" + str(
                                                                                  t) + "_" + str(k) + "_" + str(k2))
        for t in range(self.T):
            for p in range(self.P):
                for k2 in range(self.K):
                    for sw in range(self.scenario):
                        self.q[p, t, k2, sw] = self.model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS,
                                                                 name="q_" + str(p) + "_" + str(t) + "_" + str(k2))
                        self.q2[p, t, k2, sw] = self.model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS,
                                                                  name="q2_" + str(p) + "_" + str(t) + "_" + str(k2))
                        self.model.addGenConstrMax(self.q[p, t, k2, sw], [self.q2[p, t, k2, sw], 0])
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

        obj = LinExpr(0)
        obj2 = LinExpr(0)
        obj3 = LinExpr(0)
        for p in range(self.P):
            obj.addTerms(self.F[p], self.y[p])
            for k in range(self.K):
                obj.addTerms(self.C[p, k], self.x[p, k])
            for t in range(self.T):
                for sw in range(self.scenario):
                    obj2.addTerms(1, self.z[p, t, sw])
        for t in range(self.T):
            for p in range(self.P):
                for k2 in range(self.K):
                    for sw in range(self.scenario):
                        obj3.addTerms(1, self.q[p, t, k2, sw])

        if self.objtype == 0:  # min cost but big area is meet
            self.model.setObjective(obj + obj2, GRB.MINIMIZE)  ##obj1
        elif self.objtype == 1:  # max area demand but don't care cost:
            self.model.setObjective(obj + obj2 + 10000 * obj3, GRB.MINIMIZE)  ##obj2
        elif self.objtype == 2:  # balance
            self.model.setObjective(obj + obj2 + 50 * obj3, GRB.MINIMIZE)  ##obj3 where 50是一个惩罚因子，越大越说明cover demand越重要

        lhs = LinExpr(0)
        for p in range(self.P):
            lhs.addTerms(1, self.y[p])
        self.model.addConstr(lhs <= self.maxnumber)

        self.model.addConstr(obj + 1 / self.scenario * obj2 <= self.budget)
        # 有效不等式开始
        for t in range(self.T):
            for sw in range(self.scenario):
                for a in range(self.A):
                    for k2 in range(self.K):
                        lhs = LinExpr(0)
                        for p in self.regions[a]:
                            for k in range(self.K):
                                if k <= k2:
                                    lhs.addTerms(self.N[k] * (1 - self.alpha[p, t, k]), self.x[p, k])
                        self.model.addConstr(lhs >= (self.D[a, t, k2, sw] - self.B_a[a, k2]),
                                             name="fuck1")  # constraint 0

        for t in range(self.T):
            for sw in range(self.scenario):
                for a in range(self.A):
                    lhs = LinExpr(0)
                    rhs = 0
                    for p in self.regions[a]:
                        for k in range(self.K):
                            lhs.addTerms(self.N[k] * (1 - self.alpha[p, t, k]), self.x[p, k])
                    for k in range(self.K):
                        rhs += (self.D[a, t, k, sw] - self.B_a[a, k])
                    self.model.addConstr(lhs >= rhs, name=f"fuck2_{a}")  # constraint 1

        # for t in range(self.T):
        #     for sw in range(self.scenario):
        #         for p in range(self.P):
        #             for k2 in range(self.K):
        #                 lhs = LinExpr(0)
        #                 for k in range(self.K):
        #                     if k <= k2:
        #                         lhs.addTerms(self.N[k] * (1 - self.gamma[p, t, k] * self.alpha[p, t, k]),
        #                                      self.x[p, k])
        #                 self.model.addConstr(
        #                     lhs >= self.gamma[p, t, k2] * self.D_p[p, t, k2, sw] - self.B[p, k2])  # constraint 2
        #
        # for t in range(self.T):
        #     for sw in range(self.scenario):
        #         for p in range(self.P):
        #             lhs = LinExpr(0)
        #             rhs = 0
        #             for k in range(self.K):
        #                 lhs.addTerms(self.N[k] * (1 - self.gamma[p, t, k] * self.alpha[p, t, k]), self.x[p, k])
        #                 rhs += self.gamma[p, t, k] * self.D_p[p, t, k, sw] - self.B[p, k]
        #             self.model.addConstr(lhs >= rhs)  # constraint 2
        # 有效不等式结束【后2个不要了，现在变成软约束】

        for t in range(self.T):
            for sw in range(self.scenario):
                for a in range(self.A):
                    for k2 in range(self.K):
                        lhs = LinExpr(0)
                        for p in self.regions[a]:
                            for k in range(self.K):
                                if k <= k2:
                                    lhs.addTerms(1 - self.alpha[p, t, k], self.w_k[p, t, k, k2, sw])
                        self.model.addConstr(lhs + self.B_a[a, k2] >= self.D[a, t, k2, sw],
                                             name="area demand cons")  # constraint 1

        for t in range(self.T):
            for sw in range(self.scenario):
                for p in range(self.P):
                    for k2 in range(self.K):
                        lhs = LinExpr(0)
                        for k in range(self.K):
                            if k <= k2:
                                lhs.addTerms(1, self.w_k[p, t, k, k2, sw])
                        self.model.addConstr(self.q2[p, t, k2, sw] >= -lhs - self.B[p, k2] + self.gamma[p, t, k2] * (
                                self.D_p[p, t, k2, sw] + self.alpha[p, t, k2] * lhs))  # constraint 修改版

        for p in range(self.P):
            for sw in range(self.scenario):
                lhs = LinExpr(0)
                for k in range(self.K):
                    lhs.addTerms(1, self.x[p, k])
                self.model.addConstr(lhs <= self.U[p] * self.y[p])  # constraint 3
                self.model.addConstr(self.y[p] <= lhs)  # constraint 4
                for t in range(self.T):
                    self.model.addGenConstrPWL(self.w[p, t, sw], self.z[p, t, sw], [0, 1000, 2000, 3000],
                                               [0, 100, 300, 600],  # NOTE: 电价在这结果不好就挑他的锅！！这里要调大了
                                               "myPWLConstr")  # constraint 6

        # 手动线性化f_{pt}\left( w_{pt} \right) =w_{pt}\lambda _1+\left( 5w_{pt}\lambda _2+100\lambda _2 \right) +\left( 10w_{pt}\lambda _3+600\lambda _3 \right)
        # self.wz_f = {}
        # self.wz_ff = {}
        # for p in range(self.P):
        #     for t in range(self.T):
        #         for this in range(3):
        #             self.wz_f[p, t, this] = self.model.addVar(vtype=GRB.BINARY,
        #                                                       name="wzf_" + str(p) + str(t) + str(this))
        #             self.wz_ff[p, t, this] = self.model.addVar(lb=0, ub=10000000, vtype=GRB.CONTINUOUS,
        #                                                        name="wzff_" + str(p) + str(t) + str(this))

        # self.model.addConstr(self.z[p, t]==self.wz_f[p,t,0]*(self.w[p, t])+self.wz_f[p,t,1]*(5*self.w[p, t]+100)+self.wz_f[p,t,2]*(10*self.w[p, t]+600))

        # self.model.addConstr(self.z[p, t] == self.wz_ff[p, t, 0] + 5*self.wz_ff[p, t, 1] + self.wz_f[p, t,1] * 100+ self.wz_ff[p, t, 2] * 10+ 600*self.wz_f[p, t, 2] )
        # for this in range(3):
        #     self.model.addConstr(self.wz_ff[p, t, this] <= 100000*self.wz_f[p,t,this])
        #     self.model.addConstr(self.wz_ff[p, t, this] <=self.w[p,t])
        #     self.model.addConstr(self.wz_ff[p, t, this] >= self.w[p, t]-100000*(1-self.wz_f[p,t,this]))
        #
        # self.model.addConstr(self.wz_f[p,t,0]+self.wz_f[p,t,1]+self.wz_f[p,t,2]==1)
        # self.model.addConstr(self.w[p, t]>=0)
        # self.model.addConstr(self.w[p, t] <=100000*(1-self.wz_f[p,t,0])+100)
        # self.model.addConstr(self.w[p, t] >= self.wz_f[p, t, 1]* 100)
        # self.model.addConstr(self.w[p, t] <= 100000*(1-self.wz_f[p,t,1])+200)
        # self.model.addConstr(self.w[p, t] >= self.wz_f[p, t, 2] * 200)
        # self.model.addConstr(self.w[p, t] <= 100000 * (1 - self.wz_f[p, t, 2]) + 300)
        for p in range(self.P):
            for sw in range(self.scenario):
                for t in range(self.T):
                    for k in range(self.K):
                        llhs = LinExpr()
                        for k2 in range(self.K):
                            if k <= k2:
                                llhs.addTerms(1, self.w_k[p, t, k, k2, sw])
                        self.model.addConstr(llhs <= self.N[k] * self.x[p, k], name="cons7_")  # constraint 7

        for t in range(self.T):
            for sw in range(self.scenario):
                for p in range(self.P):
                    lhs = LinExpr()
                    for k in range(self.K):
                        for k2 in range(self.K):
                            if k <= k2:
                                lhs.addTerms(1, self.w_k[p, t, k, k2, sw])
                    self.model.addConstr(self.w[p, t, sw] == self.D_one * lhs)

        # model.computeIIS()

        self.model.write("chargermodel0731.lp")
        self.model.Params.MIPGap = 0.0001
        self.model.Params.TimeLimit = self.timelimit
        self.model.optimize()

        if self.model.status == GRB.INFEASIBLE:

            # self.model.computeIIS()
            # self.model.write("chargermodel.ilp")
            self.finalresult = str("No solution")
            ###这里最终版返回一个语句，不需要iis
            return 0, 0, 0, 0, self.finalresult

        else:
            for p in range(self.P):
                for k in range(self.K):
                    self.finalresult[p, k] = self.x[p, k].x

            # print("fixed cost:", obj.getValue(), "operations cost", obj2.getValue(),"uncover demand:",obj3.getValue())
        #

        return obj.getValue() + obj2.getValue(), obj.getValue(), obj2.getValue(), obj3.getValue(), self.finalresult
    def create_map(self, locations_and_sizes, df_b):
        # 创建基础地图（以香港为例）
        base_map = folium.Map(location=[22.3193, 114.1694], zoom_start=12)

        # 遍历每个被选中的地点及其规模
        for location_info in locations_and_sizes:
            location_index, size = location_info

            # 使用location_index在B文件中查找地点的经纬度
            location_data = df_b[df_b['id'] == location_index]
            if not location_data.empty:
                lat = location_data.iloc[0]['Latitude']
                lng = location_data.iloc[0]['Longitude']

                # 根据规模大小设置标记的颜色或大小
                if size > 5:
                    color = 'red'
                    radius = 10
                elif size > 1:
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
                    popup=f'Size: {size} car spaces'
                ).add_to(base_map)

        return base_map

    import pandas as pd
    import gurobipy as gp
    from gurobipy import GRB

    import pandas as pd
    import gurobipy as gp
    from gurobipy import GRB

    def optimize_charging_stations(budget, max_stations, objective_type_value, time_limit, demand_file_path,
                                   parameter_file_path):
        try:
            # 调试日志
            print("Starting optimization process with the following parameters:")
            print(
                f"Budget: {budget}, Max Stations: {max_stations}, Objective Type: {objective_type_value}, Time Limit: {time_limit}")
            print(f"Demand file: {demand_file_path}, Parameter file: {parameter_file_path}")

            # 加载需求数据和参数数据
            demand_df = pd.read_csv(demand_file_path)
            parameter_df = pd.read_csv(parameter_file_path)

            print("Demand and parameter data loaded successfully.")

            # 数据处理和填充缺失值
            parameter_df['Average_Property_Rent'].fillna(parameter_df['Average_Property_Rent'].mean(), inplace=True)

            # 创建优化模型
            model = gp.Model("charging_station_optimization")
            model.setParam('TimeLimit', time_limit)

            # 定义变量
            station_vars = model.addVars(len(parameter_df), vtype=GRB.BINARY, name="station")

            # 设定预算约束
            model.addConstr(
                gp.quicksum(station_vars[i] * parameter_df.loc[i, 'Cost'] for i in range(len(parameter_df))) <= budget,
                "budget_constraint")

            # 限制最大站点数量
            model.addConstr(gp.quicksum(station_vars) <= max_stations, "station_limit")

            # 根据objective_type_value选择不同的目标函数
            if objective_type_value == 0:  # 最小化成本
                model.setObjective(
                    gp.quicksum(station_vars[i] * parameter_df.loc[i, 'Cost'] for i in range(len(parameter_df))),
                    GRB.MINIMIZE)
            elif objective_type_value == 1:  # 最大化需求覆盖
                model.setObjective(
                    gp.quicksum(station_vars[i] * demand_df.loc[i, 'Demand'] for i in range(len(demand_df))),
                    GRB.MAXIMIZE)
            else:  # 平衡需求和成本
                model.setObjective(gp.quicksum(
                    station_vars[i] * (demand_df.loc[i, 'Demand'] - parameter_df.loc[i, 'Cost']) for i in
                    range(len(demand_df))), GRB.MAXIMIZE)

            print("Objective function set, starting optimization...")

            # 优化模型
            model.optimize()

            print(f"Optimization completed. Status: {model.Status}")

            if model.Status != GRB.OPTIMAL:
                raise RuntimeError("Optimization did not converge to an optimal solution.")

            # 提取优化结果
            result = {
                'TotalCost': model.objVal,
                'FixedCost': sum(
                    parameter_df.loc[i, 'Cost'] for i in range(len(parameter_df)) if station_vars[i].x > 0.5),
                'OperationsCost': model.objVal - sum(
                    parameter_df.loc[i, 'Cost'] for i in range(len(parameter_df)) if station_vars[i].x > 0.5),
                'UncoveredDemand': sum(
                    demand_df.loc[i, 'Demand'] for i in range(len(demand_df)) if station_vars[i].x < 0.5),
                'FinalResult': [
                    {"latitude": parameter_df.loc[i, 'Latitude'], "longitude": parameter_df.loc[i, 'Longitude']} for i
                    in range(len(parameter_df)) if station_vars[i].x > 0.5]
            }

            print("Optimization results extracted successfully.")
            return result

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            raise RuntimeError(f"Optimization failed: {str(e)}")


# import pandas as pd
# import numpy as np
# import gurobipy as gp
# from gurobipy import GRB
#
#
# class ChargingStationOptimizer:
#     def __init__(self, demand_file_path, parameter_file_path, input_budget, input_maxstation, input_objtype,
#                  input_timelimit):
#         self.demand_file_path = demand_file_path
#         self.parameter_file_path = parameter_file_path
#         self.budget = input_budget
#         self.maxnumber = input_maxstation
#         self.objtype = input_objtype
#         self.timelimit = input_timelimit
#         self.read_instance()
#
#     def read_instance(self):
#         try:
#             demand_df = pd.read_csv(self.demand_file_path)
#             parameter_df = pd.read_csv(self.parameter_file_path)
#
#             print("Demand DataFrame head:")
#             print(demand_df.head())
#             print("\nParameter DataFrame head:")
#             print(parameter_df.head())
#
#             # 检查并处理 NaN 值
#             if demand_df.isnull().values.any():
#                 print("Warning: NaN values found in demand data. Filling with 0.")
#                 demand_df = demand_df.fillna(0)
#
#             if parameter_df.isnull().values.any():
#                 print("Warning: NaN values found in parameter data. Filling with mean values.")
#                 parameter_df = parameter_df.fillna(parameter_df.mean())
#
#             self.T = 24  # Period
#             self.A = 18  # Number of areas
#             self.P = len(parameter_df)  # Number of candidate stations
#
#             # 处理地区名称
#             district = ["KOWLOON CITY", "YAU TSIM MONG", "SHAM SHUI PO", "KWUN TONG", "WONG TAI SIN", "EASTERN",
#                         "CENTRAL AND WESTERN", "SOUTHERN", "SAI KUNG", "SHA TIN", "TUEN MUN", "TSUEN WAN", "YUEN LONG",
#                         "TAI PO", "NORTH", "KWAI TSING", "ISLAND", "WAN CHAI"]
#             district_to_id_dict = {d: i for (i, d) in enumerate(district)}
#
#             # 确保 'Average_Property_Rent' 列存在并且没有 NaN 值
#             if 'Average_Property_Rent' not in parameter_df.columns:
#                 raise ValueError("'Average_Property_Rent' column not found in parameter file.")
#
#             self.F = parameter_df["Average_Property_Rent"].values / 30  # Building cost
#
#             # 检查 F 中是否有 NaN 或 Inf 值
#             if np.isnan(self.F).any() or np.isinf(self.F).any():
#                 print("Warning: NaN or Inf values found in F. Replacing with mean value.")
#                 mean_F = np.nanmean(self.F)
#                 self.F = np.where(np.isnan(self.F) | np.isinf(self.F), mean_F, self.F)
#
#             # Initialize D_p (demand)
#             self.D_p = np.zeros((self.P, self.T, 2, 1))  # Assuming 2 types of chargers and 1 scenario
#             for index, row in demand_df.iterrows():
#                 p = row.get('id', 0)  # 假设 'id' 列代表充电站 ID
#                 t = row.get('Hour', 0)  # 假设 'Hour' 列代表时间
#                 demand = row.get('Total Demand Count', 0)
#                 if p < self.P and t < self.T:
#                     self.D_p[p, t, 0, 0] = demand * 0.5  # Slow charging demand
#                     self.D_p[p, t, 1, 0] = demand * 0.5  # Fast charging demand
#
#             print(f"Data loaded successfully. P={self.P}, T={self.T}, A={self.A}")
#             print(f"F shape: {self.F.shape}, D_p shape: {self.D_p.shape}")
#
#         except Exception as e:
#             print(f"Error in read_instance: {str(e)}")
#             raise
#
#     @classmethod
#     def optimize_charging_stations(cls, budget, max_stations, objective_type_value, time_limit, demand_file_path,
#                                    parameter_file_path):
#         try:
#             optimizer = cls(demand_file_path, parameter_file_path, budget, max_stations, objective_type_value,
#                             time_limit)
#
#             # Create optimization model
#             model = gp.Model("charging_station_optimization")
#             model.setParam('TimeLimit', optimizer.timelimit)
#
#             # Define variables
#             station_vars = model.addVars(optimizer.P, vtype=GRB.BINARY, name="station")
#
#             # Set budget constraint
#             model.addConstr(
#                 gp.quicksum(station_vars[i] * optimizer.F[i] for i in range(optimizer.P)) <= optimizer.budget,
#                 "budget_constraint")
#
#             # Limit maximum number of stations
#             model.addConstr(gp.quicksum(station_vars) <= optimizer.maxnumber, "station_limit")
#
#             # Set objective function based on objective_type_value
#             if optimizer.objtype == 0:  # Minimize cost
#                 model.setObjective(gp.quicksum(station_vars[i] * optimizer.F[i] for i in range(optimizer.P)),
#                                    GRB.MINIMIZE)
#             elif optimizer.objtype == 1:  # Maximize demand coverage
#                 model.setObjective(gp.quicksum(station_vars[i] * np.sum(optimizer.D_p[i]) for i in range(optimizer.P)),
#                                    GRB.MAXIMIZE)
#             else:  # Balance demand and cost
#                 model.setObjective(gp.quicksum(
#                     station_vars[i] * (np.sum(optimizer.D_p[i]) - optimizer.F[i]) for i in range(optimizer.P)),
#                                    GRB.MAXIMIZE)
#
#             # Optimize model
#             model.optimize()
#
#             if model.Status != GRB.OPTIMAL:
#                 raise RuntimeError("Optimization did not converge to an optimal solution.")
#
#             # Extract optimization results
#             result = {
#                 'TotalCost': model.objVal,
#                 'FixedCost': sum(optimizer.F[i] for i in range(optimizer.P) if station_vars[i].x > 0.5),
#                 'OperationsCost': model.objVal - sum(
#                     optimizer.F[i] for i in range(optimizer.P) if station_vars[i].x > 0.5),
#                 'UncoveredDemand': sum(np.sum(optimizer.D_p[i]) for i in range(optimizer.P) if station_vars[i].x < 0.5),
#                 'FinalResult': [{'station_id': i, 'selected': station_vars[i].x > 0.5} for i in range(optimizer.P)]
#             }
#
#             return result
#         except Exception as e:
#             print(f"Error during optimization: {str(e)}")
#             raise RuntimeError(f"Optimization failed: {str(e)}")
#





if __name__ == '__main__':
    LBBD_solver = LBBD()
    # LBBD_solver.set_instance()
    demand_file_path = './A_update.csv'
    parameter_file_path = './B_update.csv'
    LBBD_solver.read_instance(demand_file_path, parameter_file_path, 100000000, 100, 2, 60)
    # input_budget,input_maxstation,input_objtype,input_timelimit,
    # where input_objtype is {0,1,2} represent
    # 0: min total cost when area demand is meet but demand of each p is ignored;
    # 1: max cover demand of each p and rea demand is meet, then to min cost
    # 2: min total cost when area demand is meet but demand of each p is moderate consideration;
    totalcost, fixcost, operationscost, uncoverddemand, finalresult = LBBD_solver.build_MIP_model()
    print("totalcost:", totalcost, "fixcost:", fixcost, "operationscost", operationscost, "uncoverdemand:",
          uncoverddemand)
    print("location and size:", finalresult)
