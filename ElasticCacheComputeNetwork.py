"""
 A packet level simulator for Elastic Caching + Service Chain Computing Network. By Jinkun Zhang, Dec 2023.
 The network jointly optimizes routing/offloading and caching strategy with congestion-dependent cost and arbitrary topology.
 The cache sizes are also jointly determined, to minimize a network-wide cost function.
 This is PULL-BASED system. Request pulls computation results, and computation demand pulls previous stage resutls (or raw data).
 """

from array import array
from cmath import inf, isfinite, isinf, isnan
import logging,argparse
from queue import Empty
import time
from tkinter import N
from turtle import forward
import simpy
#from networkx import Graph, DiGraph,shortest_path
import networkx as nx
import numpy as np
import scipy.optimize as sciopt
import matplotlib.pyplot as plt
import pickle
import os
import random
import sys
import heapq
from collections import Counter
from cvxopt import solvers
from cvxopt import matrix

LineBreak = '---------------------------------------------------------\n'
PriorityCacheType = ['LRU','LFU','FIFO']        # priority cache names, will be realized according to a priority list related to previously visited items
OptimizedCacheType = ['GP','LMIN','GCFW','Continuous-Greedy','Greedy','CostGreedy','AC-R','MinDelay','AC-N']              # optimized cache names, will be realized by random cache according to a continuous caching variable
OtherCacheType = ['RR']                         # other cache types

eps = 1e-3

class SimulationConfig:
    def __init__(self,T_sim,T_slot,L_update,T_monitor,Is_draw_network,MonitorInitTime,T_IDReset,T_extension,\
        RequestGenerateType, RequestDelayType, RequestDelay, RequestSize,\
        ResponseDelayType, ResponseDelay, ControlDelay, ControlSize):
        self.T_sim = T_sim  # total simulation time
        self.T_slot = T_slot    # duration of each slot
        self.L_update = L_update    # number of slots between two update slots
        self.T_monitor = T_monitor  # duation of monotor period. Cost are calculated from flows averaged each monitor period
        self.Is_draw_network = Is_draw_network  # if True, draw a network figure at each monitor time
        self.MonitorInitTime = MonitorInitTime  # initial time for monitor
        self.T_IDReset = T_IDReset  # ID number reset time
        self.T_extension = T_extension  # Runtime after T_sim finished. Used to wait for un-satsified requests, no new requests are generated.

        self.RequestGenerateType = RequestGenerateType  # request generating type, poisson or uniform
        self.RequestDelayType = RequestDelayType
        self.RequestDelay = RequestDelay
        self.RequestSize = RequestSize
        self.ResponseDelayType = ResponseDelayType
        self.ResponseDelay = ResponseDelay
        #self.ResponseSize = ResponseSize
        self.ControlDelay = ControlDelay
        self.ControlSize = ControlSize

class AlgorithmConfig:
    def __init__(self,CacheAlgo,RouteAlgo,SizeAlgo,StepsizeGP_phi,StepsizeGP_y,Thrt,N_GCFW,RouteInitType,AllowLoop,RouteType):
        self.CacheAlgo = CacheAlgo
        self.RouteAlgo = RouteAlgo
        self.SizeAlgo = SizeAlgo
        #self.CacheTypeAlgo = CacheTypeAlgo
        self.StepsizeGP_phi = StepsizeGP_phi
        self.StepsizeGP_y = StepsizeGP_y
        self.Thrt = Thrt 
        self.N_GCFW = N_GCFW
        self.RouteInitType = RouteInitType
        self.AllowLoop = AllowLoop 
        self.RouteType = RouteType

def testLP():
    """ Test LP for large dimension. """
    print('Runing Test LP')
    R = 100
    P = 3
    V = 50
    I = 50
    var_dim = R * P * V * I
    const_dim = V + R*P + var_dim
    w = np.random.rand(var_dim)
    A = np.random.rand(const_dim,var_dim)
    b = np.random.rand(const_dim)
    sciopt.linprog(w,A,b)

def PorjectOnSimplex(y,Cap = 1):
    """ project variable y onto simplex : \sum_i y_i = Cap. 
    use the method from https://math.stackexchange.com/questions/3778014/matlab-python-euclidean-projection-on-the-simplex-why-is-my-code-wrong. """
    dim = len(y)
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - Cap
    ind = np.arange(dim) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(y - theta, 0)
    return w

def projectToSimplex_UpperBoundBy1_legacy(y,Cap):
    """ project variable y onto simplex : \sum_i y_i = Cap. But also require y_i <=1.
    (Legacy method, by solving QP)"""
    #keys, vals = zip( *[ (key,d[key]) for key in d] )
    n = len(y)
    q = -matrix(y)
    #print('q = '+str(q)+', type: '+str(type(q)))
    P = matrix(np.eye(n))

    G = matrix(np.concatenate( (np.eye(n), -np.eye(n), np.ones((1,n)) ) ))
    
    h = matrix( n*[1.0] + n*[0.0] +[Cap]  )   
     
    solvers.options['show_progress'] = False
    res = solvers.qp(P,q,G,h)
   
    sol = res['x']
    return sol

def projectToSimplex_UpperBoundBy1(y,Cap):
    """ project variable y onto simplex : \sum_i y_i = Cap. But also require y_i <=1.
    (New method, by Volkan)"""
    return projectToSimplex_UpperBoundBy1_legacy(y,Cap)
    dim = len(y)
    if dim == 0:
        return np.array([])
    elif dim == 1:
        return np.minimum( 1, np.maximum(0, y) )
    
    if min(y) >= 0.0 - eps and max(y) <= 1.0 + eps and sum(y) <= Cap + eps:
        return np.minimum( 1, np.maximum(0, y) )

    mu = sorted(y, reverse=True)    # sort y in descending order

    # then find two index i<j in 1,2,...,dim with maximum distance |j-i|, and with the following hold:
    # mu_i - s < 1 and mu_j - s > 0
    # where s = 1 / (j-i+1) * (\sum_{r = i}^j mu_r - Cap + i - 1)

    # to find such (i,j), we search for all (i,j) with distance from large to small
    # that is, totally dim-1 steps. First step check (1,dim), second check (1,dim-1) and (2,dim).
    # at k-th step, check k pairs (1,dim-k+1) to (k,dim) with distance dim-k. Until finish

    for k in range(1,dim):      # here k has value 1 to dim-1
        check_pairs = [(t,dim-k+t) for t in range(1,k+1)]   # here t = 1 to k, corresponding to pair (t, dim-k+t)
        for (i,j) in check_pairs:
            s = 1.0 / (j-i+1) * ( sum([mu[r-1] for r in range(i,j+1)]) -Cap + i -1 )    # here r = i to j, so mu[r-1] is the r-th element
            if mu[i-1] - s < 1.0 and mu[j-1] - s > 0.0:
                opt_pair = (i,j)

    # after find such (sigma,rho) = (i,j),  let theta = 1 / (rho-sigma+1) * (\sum_{i=rho}^sigma mu_i - Cap + sigma -1), essentially s.
                theta = s

    # then, for each element in v, if  1>= v_i - theta >= 0, w_i = v_i - theta. Else, w_i = 0 or 1
                w = np.minimum( 1, np.maximum(0, y - theta) )
                return w

    #if np.count_nonzero( np.maximum(y - 0.1,0)) <= Cap:     # if the element larger than 0.1 is less than cap, strip these snmall elements
    #    return np.minimum( 1, np.maximum(0, ArrayStrip(y,0.1)) )

    #print('ERROR: Fail to project! y = '+str(y)+', Cap = '+str(Cap))
    #print('min(y) = '+str(min(y))+', max(y) = '+str(max(y))+', sum(y) = '+str(sum(y)))
    return projectToSimplex_UpperBoundBy1_legacy(y,Cap)
    exit()

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def Zipf(a: np.float64, min: np.uint64, max: np.uint64, size=None):
    """
    Generate Zipf-like random variables,
    but in inclusive [min...max] interval
    """
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(min, max+1) # values to sample
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)            # normalized

    return np.random.choice(v, size=size, replace=True, p=p)

def CopyRoutVar(RoutVar):
    """ Deep copy of the input routing var.
    Input: dict of dict { node i : item k: [array of phi_ijk for all j]} """
    RoutVar_copy = {}
    for node in RoutVar:
        RoutVar_copy[node] = {}
        for item in RoutVar[node]:
            RoutVar_copy[node][item] = RoutVar[node][item].copy()
    return RoutVar_copy

class Message():
    """A general message object."""
    def __init__(self,MsgType,ID,Source,Item,Stage,Payload,Size):
        self.MsgType = MsgType  # Type of this message, could be 'Request','Response','Control'
        self.ID = ID            # Msg id. Used to identify different instance of same requester and item
        self.Source = Source    # Source node of the msg. If request, the original requester. If response, also the requester. If control, equal to the sender
        self.Sender = None      # Sender of the msg, will change while traveling
        self.Item = Item        # Item that are requested. If is control msg, item = None
        self.Stage = Stage      # Request for a paticual stage for Item, or carries the stage. If control msg, stage = None
        self.Payload = Payload  # Payload used for control message, and aggregated link cost for response msg
                                # for ctrl msg, payload = (float1,float2,bool)
                                #   float1:   carrying pTpr_j,
                                #   float2:   initialized to None, will be set to D'ji(Fji) when passing through (j,i)
                                #   bool:     marker of exsitence of "improper link" from j downstream
                                # for response msg, payload = (int,float)
                                #   int: number of hops traveled since generated
                                #   float: aggregated link costs along the response path. Evaluated with marginal cost at F=0
        self.Size = Size        # Size of this msg, set by configuration
    
    def __str__(self):  # use to print
        return 'Msg: '+self.MsgType+', Source:'+str(self.Source)+ ', Item:'+str(self.Item)+', Sender:'+str(self.Sender)+', Payload:'+str(self.Payload)+', Size:'+str(self.Size)

class SimpleCache():
    """ A simple cache object with function of insert, delete, look up, and status check."""
    def __init__(self,_id) -> None:
        self._id = _id
        #self.M = M
        self.size = 0
        self._content = []
    def setSize(self,targetSize):   # target size is a integer
        self.size = targetSize
    def insert(self,item):   
        if item not in self._content:
            self._content.append(item)
    def delete(self,item):                  
        if item in self._content: self._content.remove(item) 
    def ContainItem(self,item):
        return item in self._content
    def PickItem(self,pos):
        return self._content[pos] if pos < self.OccupiedSize()  else None
    def OccupiedSize(self):
        return len(self._content)
    def GetSize(self):
        return self.size

class LRUCache(SimpleCache):
    """ LRU cache, maintains a least recent use queue in self._content. """
    def __init__(self,_id) -> None:
        super().__init__(_id)
    def CacheCheck(self,item):
        """ LRU cache check, returns if item is in the cache, and automatically update cache decision. """
        if self.ContainItem(item):          # if is cached, put item in the tail of the queue and return true
            self.delete(item)
            self.insert(item)
            return True
        else:                               # if not cached, check size, put item in the tail and return false
            if self.OccupiedSize() < self.size:    
                self.insert(item)
                return False
            elif self.size == 0:
                return False
            else:
                self.delete(self.PickItem(0))
                self.insert(item)
                return False

class LFUCache(SimpleCache):
    """ LFU cache, separately maintains a dict of usage frequency count. """
    def __init__(self,_id) -> None:
        super().__init__(_id)
        self.ItemFreq = Counter()
    def refresh(self):
        """ Refresh the cached items according to usage count. """
        TopItems = self.ItemFreq.most_common(int(self.size))
        self._content = [tup[0] for tup in TopItems]
    def CacheCheck(self,item):
        """ LFU cache check, return if item is in the cache, and automatically update cache decision. """
        if self.ContainItem(item):
            self.ItemFreq[item] += 1
            return True
        else:
            self.ItemFreq[item] += 1
            self.refresh()
            return False

class FIFOCache(SimpleCache):
    """ FIFO cache, maintains a fifo queue in self._content. """
    def __init__(self,_id) -> None:
        super().__init__(_id)
    def CacheCheck(self,item):
        """ FIFO cache check, returns if item is in the cache, and automatically update cache decision. """
        if self.ContainItem(item):
            return True
        else:
            if self.OccupiedSize() < self.size:    
                self.insert(item)
                return False
            elif self.size == 0:
                return False
            else:
                self.delete(self.PickItem(0))
                self.insert(item)
                return False

class RRCache(SimpleCache):
    """ Random replacement cache, randomly caches items that has been checked with. Need to manually refresh every slot."""
    def __init__(self,_id) -> None:
        super().__init__(_id)
        self.ItemList = []
    def refresh(self):
        if self.size >= self.OccupiedSize():
            self._content = self.ItemList.copy()
        else:
            self._content = np.random.choice(self.ItemList,size=int(self.size),replace=False)
    def CacheCheck(self,item):
        if item not in self.ItemList:
            self.ItemList.append(item)
        return self.ContainItem(item)

class OptCache():
    """ Optimized cache type, LMIN or GP, need exogenous refresh. No 'size' attribute. """
    def __init__(self,_id) -> None:
        self._id = _id
        self._content = []       # [list of cached items]
        self.size = 0                                # only used with fix-capacity caching algorithm (Cont-Greedy, MinDelay)
    def update(self, new_list):
        """ completely replace the content. """
        self._content = new_list.copy()
    def CacheCheck(self,item):
        if item in self._content:
            return True
        else:
            return False
    def GetSize(self):
        #print('Geting size: '+str(self._content[m]))
        return len(self._content)

def cacheGenerator(CacheType,_id):
    if CacheType == 'LRU':
        return LRUCache(_id)
    elif CacheType == 'LFU':
        return LFUCache(_id)
    elif CacheType == 'FIFO':
        return FIFOCache(_id)
    elif CacheType == 'RR':
        return RRCache(_id)
    elif CacheType in OptimizedCacheType:
        return OptCache(_id)
    else:
        print('ERROR: Cache type not defined.')
        exit()

def ArrayStrip(a,eps):
    """ Set the element smaller than eps in a to 0. """
    return [a[k] if a[k] >= eps else 0 for k in range(len(a))]

def ElasticRounding(y):
    """ Adaptive distributed rounding for elastic cache sizes. Do not consider cache types.
    y is a dict {item: stage: y_ks}.
    Returns an instance of cache dicision """
    eps = 1e-2
    Items = list(y.keys())
    Stages = list(y[Items[0]].keys())
    # Cartesian product, used to order the y vector 
    prod = [(k,s) for k in Items for s in Stages]
    K = len(prod)
    #print('prod = '+str(prod))
    #print('y = '+str(y))
    y_vec = [y[p[0]][p[1]] for p in prod]

    y_strip = ArrayStrip(y_vec,eps)    # delete small values in p
    bar_start = np.zeros(K)     # bar start positions 
    bar_end = np.zeros(K)       # bar end positions
    for k in range(K):
        if bar_start[k] + y_strip[k] > 1.0:
            bar_end[k] = bar_start[k] + y_strip[k] -1.0
        else:
            bar_end[k] = bar_start[k] + y_strip[k]
        if k+1 in range(K):
            bar_start[k+1] = bar_end[k]
    
    t_array = np.random.rand()
    x_out = []                      # list of cached (item, stage) tuple
    for k in range(K):
        if y_strip[k] >= eps:                   # only cache item with p >= eps
            if bar_end[k] >  bar_start[k]:  # if the bar is completely in one line
                if bar_start[k] <= t_array and bar_end[k] > t_array:
                    x_out.append(prod[k])
            else:                           # if the bar is separated into two lines
                if bar_start[k] <= t_array or bar_end[k] > t_array:
                    x_out.append(prod[k])
    return x_out
    


class ElasticCacheComputeNetwork:
    """ Main body of simulator. """
    def __init__(self,G,AppChain_Len,PacketSize,CompWorkload,DesServerList,Demands,LinkCostType,CompCostType,CacheCostType,SimConfig,AlgoConfig):
        # Environment
        self.env = simpy.Environment()  
        # Configurations
        self.SimConfig = SimConfig
        self.AlgoConfig = AlgoConfig

        # Basic Settings
        self.G = G  # the directed graph
        self.LinkCostType = LinkCostType   
        self.CompCostType = CompCostType   
        self.CacheCostType = CacheCostType    

        self.number_of_nodes = G.number_of_nodes()
        self.Nodes = {}
        for i in range(self.number_of_nodes):
            self.Nodes[i] = {}
            self.Nodes[i]['Neighbor'] = []  # neighbor of node i. Note that we assume the network is symmetric, in-neighbor and out-neighbor are same

        self.EdgesEnumerator = [(e[0],e[1]) for e in G.edges() if e[0] is not e[1]] # enumerator for all edges, order-sensitive. But the edge attributes are defined in self.Edges
        self.number_of_edges = len(self.EdgesEnumerator)
        #print(self.EdgesEnumerator)
        self.Edges = {}
        for e in self.EdgesEnumerator:
            if e[0] is e[1]:
                print('Warning: e[0] = e[1].')
                continue
            if e[0] not in self.Edges.keys():
                self.Edges[e[0]] = {} 
            if e[1] not in self.Edges[e[0]].keys():
                self.Edges[e[0]][e[1]] = {}
            self.Nodes[e[0]]['Neighbor'].append(e[1])

        #self.M = len(CacheCostType)         # number of cache types
        #self.CacheTypes = range(self.M)

        self.AppChain_Len = AppChain_Len
        self.number_of_applications = len(AppChain_Len)
        self.Stages = []                    # all flow stages
        for app in range(self.number_of_applications):
            for task in range(self.AppChain_Len[app]+1):
                self.Stages.append( (app,task) )
        self.number_of_stages = len(self.Stages)



        self.number_of_items = len(DesServerList)
        self.Items = range(self.number_of_items)
        self.DesServers = {k : [DesServerList[k]] for k in range(self.number_of_items)} # dictionary of designated servers -> {item: [list of servers]}

        self.PacketSize = PacketSize    # packet size of all flow stages, dict stage:size
        self.CompWorkload = CompWorkload    # dict stage:workload

        print('AppLen = '+str(self.AppChain_Len))
        print('Stages = '+str(self.Stages))
        print('Demands = '+str(Demands))
        print('CompWorkload = '+str(self.CompWorkload))
        print('Designated Servers = '+str(self.DesServers))

        self.RequestRates = {}  # dict of request rates -> {node:item:stage:rate}
        for i in self.Nodes:
            self.RequestRates[i] = {}
            for k in range(self.number_of_items):
                self.RequestRates[i][k] = {}
                for stage in self.Stages:
                    self.RequestRates[i][k][stage] = 0.0

        for d in Demands:
            requester = d[1]
            item = d[2]
            stage = d[3]
            rate = d[4]
            self.RequestRates[requester][item][stage] += rate

        # Record for cost calculation
        #for i in self.Nodes:
            #self.Nodes[i]['CacheSize'] = np.zeros(self.M)   # list of cache size, initialized to 0, should be integer and kept unchanged between two update slots
        for e in self.EdgesEnumerator:
            self.Edges[e[0]][e[1]]['FlowCount'] = 0.0       # sum of msg size passed through e. Note when size is large, this variable could overflow 
            self.Edges[e[0]][e[1]]['CurrentFlow'] = 0.0     # edge flow, calculated by other process using FlowCount
        for i in self.Nodes:
            self.Nodes[i]['WorkloadCount'] = 0.0
            self.Nodes[i]['CurrentWorkload'] = 0.0


        # Creat caches, initialize all caches to be empty
        for i in self.Nodes:
            #self.Nodes[i]['Caches'] = {}
            #for m in self.CacheTypes:
            #    cache_id = (i,m)
            self.Nodes[i]['Cache'] = cacheGenerator(self.AlgoConfig.CacheAlgo, i)
            #print('Node '+str(i)+': cache init to '+str(self.Nodes[i]['Cache']._content))
        
        # Initialize the routing and caching variables. Routing is init by an initializer, caching is init to all 0
        self.LinkCapacity = self.LinkCapacityCal()  # calculate an upper bound to guarantee feasibility
        self.CompCapacity = self.CompCapacityCal()  # computation workload upperbound

        self.RouteVarInit = self.RouteVarInitCal()  # calculate initial routing variable
        #print(self.RouteVarInit)

        for i in self.Nodes:                        # initilize the routing variable, cache sizes are set to 0
            self.Nodes[i]['RouteVar'] = self.RouteVarInit[i].copy()     # ['RouteVar'] is a nested dict, ordered by [node]['RouteVar'][item][stage] is an array \phiij(k,s), staring from j=0 (computation) 
            self.Nodes[i]['CacheVar'] = {}                              # ['CacheVar'] is a nested dict, ordered by [node]['CacheVar'][item][stafe] is a float y_i(k,s)
            for k in self.Items:
                self.Nodes[i]['CacheVar'][k] = {}
                for s in self.Stages:
                    self.Nodes[i]['CacheVar'][k][s] = 0.0         
                                                                        # Note: ['CacheVar'] is only used when cache type is in optimized cache types. 
                                                                        # Algorithms should guarantee that y for stage 0 is the same for all apps, as it reps the caching for raw data
        if self.Is_looped():
            print('Warning: Initial routing scheme contains loops.')
        if self.Is_exceedCapacity():
            print('Warning: Initial routing scheme exceed link capacity')
        
        # calculate the extreme point: pure caching no transmission
        print('Cost of pure caching = '+str(self.Pure_Caching_Cost()))

        # Request routing record, used to deliver the responses
        for i in self.Nodes:
            self.Nodes[i]['RequestRecord'] = {}         # record is a dictionay { (requester,item,stage,id) : node_msg_from}, for all by-passing requests
            self.Nodes[i]['ExoRequestRecord'] = {}           # only record exogenous input requests, overlap with 'RequestRecord'
            self.Nodes[i]['ComputeRequestRecord'] = {}         # pending computation requests

        # Statisics to be collected.
        self.statistics = {}
        self.statistics['GeneratedRequests'] = 0        # total number of requests that ever be generated
        self.statistics['HandledRequestMessages'] = 0   # total request messages handled by request router
        self.statistics['HandledResponseMessages'] = 0  # total response messages handled by response router
        self.statistics['RequestSatisfied'] = 0         # total number of satisfied requests
        self.statistics['RequestDelay'] = {}            # record the delay status of requests
        self.statistics['RequestDelay']['Min'] = inf
        self.statistics['RequestDelay']['Max'] = 0.0
        self.statistics['RequestDelay']['Average'] = 0.0
        self.statistics['ServerHit'] = {}                # total number of requests served by designated server
        #self.statistics['CacheHit'] = {}                # total cache hits, distinguished by cache type. not include server hit
        self.statistics['CacheHit'] = {}
        self.statistics['CacheMiss'] = {}                # total number of cache miss
        self.statistics['CacheMissCost'] = {}            # sum of cache miss cost, evaluated with marginal cost at F=0
        self.statistics['ComputeRequestRecord'] = {}
        self.statistics['ComputePerformed'] = {}
        for node in self.Nodes:
            self.statistics['ServerHit'][node] = {}
            self.statistics['CacheHit'][node] = {}
            self.statistics['CacheMiss'][node] = {}
            self.statistics['CacheMissCost'][node] = {}
            self.statistics['ComputeRequestRecord'][node] = 0
            self.statistics['ComputePerformed'][node] = 0
            for item in self.Items:
                self.statistics['ServerHit'][node][item] = 0
                self.statistics['CacheHit'][node][item] = {}
                self.statistics['CacheMiss'][node][item] = {}
                self.statistics['CacheMissCost'][node][item] = {}
                for stage in self.Stages:
                    self.statistics['CacheHit'][node][item][stage] = 0
                    self.statistics['CacheMiss'][node][item][stage] = 0
                    self.statistics['CacheMissCost'][node][item][stage] = 0
        self.statistics['GeneratedControlMessages'] = 0 # total control messages generated, if using GP
        self.statistics['MonitorTime'] = []             # list of all monitor time-points
        self.statistics['MonitorRealTime'] = []             # list of all monitor time (real time, in sec), start from the init of monitor process
        self.statistics['LinkCost_actual'] = []                # list of actual link cost at monitor time-points (calculated from time-averaged link packet counts)
        self.statistics['LinkCost_theo'] = []                # list of theoretical link cost at monitor time-points (calculated from r and phi)
        self.statistics['CompCost_actual'] = []
        self.statistics['CompCost_theo'] = []
        self.statistics['CacheCost_actual'] = []               # list of total cache cost at monitor time-points
        self.statistics['CacheCost_theo'] = []               # list of total cache cost at monitor time-points
        self.statistics['TotalCost_actual'] = []               # list of actual cost at monitor time-points
        self.statistics['TotalCost_theo'] = []                  # list of theoretical cost

        # Define pipelines
        for i in self.Nodes:
            self.Nodes[i]['RequestPipe'] = simpy.Store(self.env)    # request input queue at node
            self.Nodes[i]['ResponsePipe'] = simpy.Store(self.env)   # response input queue at node
            self.Nodes[i]['ComputeRequestPipe'] = simpy.Store(self.env)   # computation request queue (note: computation requests are pended, until response comes back)
            self.Nodes[i]['ComputePipe'] = simpy.Store(self.env)            # actual computation pipe, act as a link with delay
        for e in self.EdgesEnumerator:
            self.Edges[e[0]][e[1]]['LinkPipe'] = simpy.Store(self.env)  # transmission link. Note that all msgs are using same link

        # Start Processes
        self.env.process(self.print_time_process(self.SimConfig.T_monitor)) 
        self.env.process(self.monitor_process())
        # Request generating process, generates request msgs and push to request input queue
        for i in self.Nodes:
            for k in self.Items:
                for s in self.Stages:
                    if self.RequestRates[i][k][s] > 0:   # start request generating process for r_i(k,s) > 0
                        self.env.process(self.request_generating_process(i,k,s))
        # Request routing process, exract msgs from request input queue, can further forward, decide to compute, or generate response if stage is cached
        for i in self.Nodes:
            self.env.process(self.request_handling_process(i))
        
        # Computation request handling process, make record of computation requests that i decides to perform locally
        for i in self.Nodes:
            self.env.process(self.compute_request_handling_process(i))
        
        # Response routing process, extract msgs from response input queue and further forward accroding to RequestRecord
        for i in self.Nodes:
            self.env.process(self.response_handling_process(i))

        # Link process, extract msg from link pipe and send to corr. request input queue or response input queue.
        for i in self.Nodes:
            self.env.process(self.compute_process(i))   # compute process acts like a link
        for e in self.EdgesEnumerator:
            self.env.process(self.link_process(e))

        
            self.env.process(self.flow_calculate_process(e))

        #self.LinkCapacity = self.LinkCapacityCal()
        # Random cach shuffle for every time slot, required for certain cache types
        for i in self.Nodes:
            if self.AlgoConfig.CacheAlgo in OptimizedCacheType:
                self.env.process(self.optCache_random_shuffle_process(i))
                pass
            elif self.AlgoConfig.CacheAlgo == 'RR':
                self.env.process(self.RRCache_random_shuffle_process(i))

        # if using GCFW or GP, use message broadcasting to calculate partial derivatives
        if (self.AlgoConfig.CacheAlgo == 'GP' \
            and self.AlgoConfig.RouteAlgo == 'GP' \
            and self.AlgoConfig.SizeAlgo == 'GP')\
            or \
            (self.AlgoConfig.CacheAlgo == 'GCFW' \
            and self.AlgoConfig.RouteAlgo == 'GCFW' \
            and self.AlgoConfig.SizeAlgo == 'GCFW'):
            for i in self.Nodes:
                self.Nodes[i]['ControlPipe'] = {}
                self.Nodes[i]['trafficMarg'] = {}                                 # node input marginal pT/pt, for each item, each stage (note: for stage 0 is share for all applications, as its the raw data)
                self.Nodes[i]['MargRecord'] = {}                                # in-going record for delta_ji
                self.Nodes[i]['RequestCount'] = {}                              # counter of passing by request, used for calculate traffic
                for k in self.Items:
                    self.Nodes[i]['ControlPipe'][k] = simpy.Store(self.env) 
                    self.Nodes[i]['trafficMarg'][k] = {}                        
                    self.Nodes[i]['MargRecord'][k] = {}
                    self.Nodes[i]['RequestCount'][k] = {}                              # counter of passing by request, used for calculate traffic         
                    for s in self.Stages:
                        #self.Nodes[i]['ControlPipe'][k][s] = simpy.Store(self.env)
                        self.Nodes[i]['trafficMarg'][k][s] = 0.0
                        self.Nodes[i]['MargRecord'][k][s] = {}
                        self.Nodes[i]['RequestCount'][k][s] = 0.0                              # counter of passing by request, used for calculate traffic
                    self.env.process(self.control_generating_process(i,k))      # control process is for i,k, since different stages may interact
                    self.env.process(self.control_handling_process(i,k))

        
        # different simulation settings: 
        # 1: Proposed algorithm: gradient projection
        if self.AlgoConfig.CacheAlgo == 'GP' \
            and self.AlgoConfig.RouteAlgo == 'GP' \
            and self.AlgoConfig.SizeAlgo == 'GP':
            # Create control msg pipe. Start control msg process and calculate node input marginals. Start GP variable-updating process
            self.env.process(self.block_node_process())                         # this process uses topological sort
            for i in self.Nodes:
                self.Nodes[i]['BlockNodes'] = {}                                # blocked node set
                self.Nodes[i]['Improper'] = {}                                  # markers of 'improper link' in downstream  
                for k in self.Items:
                    self.Nodes[i]['BlockNodes'][k] = {}                         # list of block nodes, initialzied to V\N_i every update slot
                    self.Nodes[i]['Improper'][k] = {}
                    for s in self.Stages:
                        self.Nodes[i]['Improper'][k][s] = False
                        self.Nodes[i]['BlockNodes'][k][s] = [ j for j in self.Nodes if j not in self.Nodes[i]['Neighbor'] ]                         # list of block nodes, initialzied to V\N_i every update slot
                        self.env.process(self.update_process_GP(i,k,s))

        # # 2: Proposed algorithm (fixed routing special case) GCFW, combined with iterative shortest-path
        # elif self.AlgoConfig.CacheAlgo == 'GCFW' \
        #     and self.AlgoConfig.RouteAlgo == 'ShortestPath' \
        #     and self.AlgoConfig.SizeAlgo == 'GCFW'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     # Note: in this case, we assume M = 1 for now.
        #     self.env.process(self.update_process_GCFW(self.AlgoConfig.N_GCFW))

        # # 3: Baseline: Zero cache + shortest path
        # elif  self.AlgoConfig.RouteAlgo == 'ShortestPath' \
        #     and self.AlgoConfig.SizeAlgo == 'Zero'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     pass

        # #3.5: LRU/LFU/Greedy + SP + UniformCache, with M = 1
        # elif (self.AlgoConfig.CacheAlgo == 'LRU' or self.AlgoConfig.CacheAlgo == 'LFU' or self.AlgoConfig.CacheAlgo == 'Greedy') \
        #     and self.AlgoConfig.RouteAlgo == 'ShortestPath' \
        #     and self.AlgoConfig.SizeAlgo == 'Uniform'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.cacheSize_process_Uniform())
        #     if self.AlgoConfig.CacheAlgo == 'Greedy':
        #         self.env.process(self.update_process_Greedy())

        # #4: LRU/LFU + SP + MaxHit, with M = 1
        # elif (self.AlgoConfig.CacheAlgo == 'LRU' or self.AlgoConfig.CacheAlgo == 'LFU') \
        #     and self.AlgoConfig.RouteAlgo == 'ShortestPath' \
        #     and self.AlgoConfig.SizeAlgo == 'MaxHit'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.cacheSize_process_MaxHit())

        # #5: LRU/LFU + SP + MinCost, with M = 1
        # elif (self.AlgoConfig.CacheAlgo == 'LRU' or self.AlgoConfig.CacheAlgo == 'LFU') \
        #     and self.AlgoConfig.RouteAlgo == 'ShortestPath' \
        #     and self.AlgoConfig.SizeAlgo == 'MinCost'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.cacheSize_process_MinCost())

        # #(Fail) 6: Continuous Greedy (Kelly Cache) + SP + MinCost
        # elif self.AlgoConfig.CacheAlgo == 'Continuous-Greedy' \
        #     and self.AlgoConfig.RouteAlgo == 'ShortestPath' \
        #     and self.AlgoConfig.SizeAlgo == 'MinCost'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.update_process_ContinuousGreedy())
        #     self.env.process(self.cacheSize_process_MinCost())

        # #7: Greedy (Kelly Cache) + SP + MinCost
        # elif self.AlgoConfig.CacheAlgo == 'Greedy' \
        #     and self.AlgoConfig.RouteAlgo == 'ShortestPath' \
        #     and self.AlgoConfig.SizeAlgo == 'MinCost'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.update_process_Greedy())
        #     self.env.process(self.cacheSize_process_MinCost())

        # #8: CostGreedy + SP
        # # for each slot, find the (node,item) pair with max aggregated miss cost, then add the node's cache size by 1, to cache the item
        # elif self.AlgoConfig.CacheAlgo == 'CostGreedy' \
        #     and self.AlgoConfig.RouteAlgo == 'ShortestPath' \
        #     and self.AlgoConfig.SizeAlgo == 'CostGreedy'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.update_process_CostGreedy())

        # #9: Adaptive Caching + Routing (Stratis) +  Uniform
        # elif self.AlgoConfig.CacheAlgo == 'AC-R' \
        #     and self.AlgoConfig.RouteAlgo == 'AC-R' \
        #     and self.AlgoConfig.SizeAlgo == 'Uniform'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.update_process_ACR())
        #     self.env.process(self.cacheSize_process_Uniform())

        # #10: Adaptive Caching + Routing (Stratis) +  MinCost
        # elif self.AlgoConfig.CacheAlgo == 'AC-R' \
        #     and self.AlgoConfig.RouteAlgo == 'AC-R' \
        #     and self.AlgoConfig.SizeAlgo == 'MinCost'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.update_process_ACR())
        #     self.env.process(self.cacheSize_process_MinCost())

        # #11: MinDelay(Milad) +  MinCost
        # elif self.AlgoConfig.CacheAlgo == 'MinDelay' \
        #     and self.AlgoConfig.RouteAlgo == 'MinDelay' \
        #     and self.AlgoConfig.SizeAlgo == 'MinCost'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.update_process_MinDelay())
        #     self.env.process(self.cacheSize_process_MinCost())

        # #12: MinDelay(Milad) +  Uniform
        # elif self.AlgoConfig.CacheAlgo == 'MinDelay' \
        #     and self.AlgoConfig.RouteAlgo == 'MinDelay' \
        #     and self.AlgoConfig.SizeAlgo == 'Uniform'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.update_process_MinDelay())
        #     self.env.process(self.cacheSize_process_Uniform())

        # #13: Adaptive Caching with Ntework-wide constraint (Stratis) +  SP
        # elif self.AlgoConfig.CacheAlgo == 'AC-N' \
        #     and self.AlgoConfig.RouteAlgo == 'ShortestPath' \
        #     and self.AlgoConfig.SizeAlgo == 'AC-N'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.update_process_ACN())

        # #14: Alternaing method: alternatingly use ACN and shortest path to the nearest cached node
        # elif self.AlgoConfig.CacheAlgo == 'AC-N' \
        #     and self.AlgoConfig.RouteAlgo == 'ALT' \
        #     and self.AlgoConfig.SizeAlgo == 'AC-N'\
        #     and self.AlgoConfig.CacheTypeAlgo == 'All-first':
        #     self.env.process(self.update_process_ALT())

        # else:
        #     pass
        #     #print('ERROR: un-defined combination of algorithms.')
        #     #exit() 
        

    def stage_to_num(self,stage):
        """ Convert a stage tuple to a unique number """
        if stage not in self.Stages:
            print('Wrong stage for conversion')
            exit()
        else:
            sum_prev = sum([self.AppChain_Len[a]+1 for a in range(stage[0])])
            return sum_prev + stage[1]

    def request_generating_process(self,node,item,stage):
        """ Process that generates request messages. Msg are sent to node's request input queue.
            Each request is given an ID, the number of ID is repeated every T_IDReset. """
        request_id = 0
        reset_round = 0 # how many round has the id has been reset
        rate = self.RequestRates[node][item][stage]
        if rate >= eps:
            #print('Spreading requests at node '+str(node)+' for item '+str(item)+', with rate '+str(rate))
            while True:
                #ProcessType,T_IDReset 
                if self.env.now <= self.SimConfig.T_sim:
                    RequestMsg =  Message(MsgType='Request', ID=request_id, Source=node,Item=item, Stage = stage, Payload=None, Size=self.SimConfig.RequestSize)
                    yield self.Nodes[node]['RequestPipe'].put(RequestMsg)
                    msg_instance = (node,item,stage,request_id)
                    self.Nodes[node]['ExoRequestRecord'][msg_instance] = self.env.now
                    #print('Request generated : '+ str(RequestMsg))
                    request_id += 1
                    self.statistics['GeneratedRequests'] += 1

                if self.env.now / self.SimConfig.T_IDReset >= reset_round +1:   # if need reset id
                    request_id = 0

                if self.SimConfig.RequestGenerateType == 'Poisson': # if is poisson process, generate expential interval
                    yield self.env.timeout(random.expovariate(rate))
                elif self.SimConfig.RequestGenerateType == 'Uniform':   # if is constant interval
                    yield self.env.timeout(1.0/rate)
                else:    # undefined process
                    exit()
        
    def request_handling_process(self,node):
        """ Process that act as the requester router for a singe node.
            This process takes messages from the request input queue and push to an out-link (cache miss) or generate a response (cache hit).
            The out-link can be a real link, or to local computation unit.
            The detailed routing algorithm is determined by node['RouteVar'], according to algorithm in config file. """
        if self.AlgoConfig.RouteType == 'Random-Round':
            L_round = 30
            forward_token_num = {}
            token_list = {}     # list of tokens, used to randomly forward
            for item in self.Items:
                forward_token_num[item] = {}
                token_list[item] = {}
                for stage in self.Stages:
                    forward_token_num[item][stage] = [0] * (self.number_of_nodes +1)
                    token_list[item] = {}
            
        while True:
            msg = yield self.Nodes[node]['RequestPipe'].get()
            self.statistics['HandledRequestMessages'] += 1
            msg_requester = msg.Source
            msg_item = msg.Item
            msg_stage = msg.Stage
            #print("msg = "+str(msg))
            if 'RequestCount' in self.Nodes[node]:
                #print(" msg_stage = "+str(msg_stage)+", msg_item = "+str(msg_item))
                if msg_stage[1] == 0:       # when receiving raw data request, count for (0,0)
                    data_stage = (0,0)
                    self.Nodes[node]['RequestCount'][msg_item][data_stage] += 1.0
                else:
                    self.Nodes[node]['RequestCount'][msg_item][msg_stage] += 1.0
            msg_id = msg.ID
            msg_sender = msg.Sender
            msg_instance = (msg_requester,msg_item,msg_stage,msg_id)
            #print('Node ' + str(node) + ' Handling request msg : '+str(msg_instance))
            #print('Corr. server = '+str(self.DesServers[msg_item]))

            # if 
            #self.Nodes[node]['ExoRequestRecord'][msg_instance] = self.env.now

            # make record of where does this msg come from, used to route corr. response
            if msg_instance not in self.Nodes[node]['RequestRecord']:  
                if msg_sender is not None:  # if this msg is received from other nodes, record the sender.
                    self.Nodes[node]['RequestRecord'][msg_instance] = msg_sender
                else:                       # if this msg is generated by this node, record the send time
                    self.Nodes[node]['RequestRecord'][msg_instance] = self.env.now
            else:
                #print('ERROR: Time:' +str(self.env.now)+'. Record already exist. Node:'+str(node)+', request instance:'+str(msg_instance)+', record:'+str(self.Nodes[node]['RequestRecord'][msg_instance]))
                #exit()
                pass
            # first check up local cache. discard the request msg, generate a response msg and push to local response input queue
            cache_hit = False
            if (msg_stage[1] == 0) and (node in self.DesServers[msg_item]):   # if is a data request and node is the designated server
                cache_hit = True
                self.statistics['ServerHit'][node][msg_item] += 1
            else:                                                       # else: check cache
                content = (msg_item,msg_stage)
                if self.Nodes[node]['Cache'].CacheCheck(content):
                    self.statistics['CacheHit'][node][msg_item][msg_stage] += 1
                    cache_hit = True
            #print('cache_hit = '+str(cache_hit))

            # if cache hit, send response, else continue to forward
            if cache_hit:
                #print('Cache hit : msg '+str(msg_instance))
                response_payload = (0, 0.0) # will be recording (number of response hops, aggregated marginal costs)
                response_size = self.PacketSize[msg_item][msg_stage]
                ResponseMsg = Message(MsgType='Response', ID=msg_id, Source=msg_requester, Item= msg_item, Stage=msg_stage, Payload=response_payload, Size=response_size)
                yield self.Nodes[node]['ResponsePipe'].put(ResponseMsg)
            else:
                # determin the next hop according to method specified in config file
                self.statistics['CacheMiss'][node][msg_item][msg_stage] += 1

                if self.AlgoConfig.RouteType == 'Random':                   # pure random packet forwarding
                    forward_prob_ori = self.Nodes[node]['RouteVar'][msg_item][msg_stage]   # forwarding probability
                    if sum(forward_prob_ori) < eps:                         # if all phi are 0, equally forward
                        forward_prob_neighbor = np.array([1.0 if j in self.Nodes[node]['Neighbor'] else 0.0 for j in self.Nodes])
                        if msg_stage[1] == 0:
                            forward_prob_ori = np.insert(forward_prob_neighbor,0,0.0,axis=0)    # put computation into position 0 with value 0.0
                        else:
                            forward_prob_ori = np.insert(forward_prob_neighbor,0,1.0,axis=0)    # put computation into position 0 with value 1.0
                    forward_prob = forward_prob_ori / sum(forward_prob_ori)
                    if np.isnan(forward_prob).any():
                        #print('forward_prob = '+str(forward_prob))
                        #print('self.Nodes[node][RouteVar] = '+str(self.Nodes[node]['RouteVar'][msg_item][msg_stage]))
                        pass
                    next_hop = np.random.choice(range(self.number_of_nodes +1), p = forward_prob)

                elif self.AlgoConfig.RouteType == 'Random-Round':
                    #round phi_ij(k) to 100 sections. Randomly choose 1 section and forward, and delete that section, 
                    #untill run off all section, then repeat. 
                    #(This will guarantee a relatively stable link flow)
                    if sum(forward_token_num[msg_item][msg_stage]) == 0:
                        # if all tokens are out, re-fill token table using current phi variable
                        forward_prob_ori = self.Nodes[node]['RouteVar'][msg_item][msg_stage]
                        if sum(forward_prob_ori) < eps:                         # if all phi are 0, equally forward
                            forward_prob_neighbor = np.array([1.0 if j in self.Nodes[node]['Neighbor'] else 0.0 for j in self.Nodes])
                            if msg_stage[1] == 0:
                                forward_prob_ori = np.insert(forward_prob_neighbor,0,0.0,axis=0)    # put computation into position 0 with value 0.0
                            else:
                                forward_prob_ori = np.insert(forward_prob_neighbor,0,1.0,axis=0)    # put computation into position 0 with value 1.0
                        forward_prob = forward_prob_ori / sum(forward_prob_ori)
                        forward_token_count = np.floor(forward_prob * L_round)
                        for j in range(len(forward_token_count)):
                            forward_token_num[msg_item][msg_stage][j] = int(forward_token_count[j])
                        if sum(forward_token_num[msg_item][msg_stage]) == 0:
                            print('ERROR: no forwarding token after re-fill, sum_phi = '+str(sum(self.Nodes[node]['RouteVar'][msg_item][msg_stage])))
                            print('y_iks = '+str(sum(self.Nodes[node]['CacheVar'][msg_item][msg_stage])))
                            print('CacheContent = '+str(self.Nodes[node]['Cache']._content))
                            exit()
                            print('Will pick the largest node to forward, 1 token')
                            prob_max = 0.0
                            max_node = self.Nodes[node]['Neighbor'][0]
                            for node_j in self.Nodes[node]['Neighbor']:
                                if self.Nodes[node]['RouteVar'][msg_item][node_j] > prob_max:
                                    prob_max = self.Nodes[node]['RouteVar'][msg_item][node_j]
                                    max_node = node_j
                            forward_token_num[msg_item][node_j] += 1

                        #print('forward_token_count = '+str(forward_token_count))
                        #print('forward_token_num[msg_item][msg_stage][j] = '+str(forward_token_num[msg_item][msg_stage][j]))
                        token_list[msg_item][msg_stage] = []
                        for j in range(len(forward_token_count)):
                            for counter_token in range(forward_token_num[msg_item][msg_stage][j]):
                                token_list[msg_item][msg_stage].append(j)
                        #print('stage = '+str(stage)+'; token_list = '+str(token_list[msg_item][msg_stage]))
                        #token_list[msg_item][msg_stage] = [j for j in range(len(forward_token_count)) for counter_token in range(forward_token_num[msg_item][msg_stage][j])]                    

                    # randomly choose next hop
                    #forward_prob_token = np.array(forward_token[msg_item]) / sum(forward_token[msg_item])
                    #next_hop = np.random.choice(range(self.number_of_nodes), p = forward_prob_token)   # this line will ocasionally fail
                    # another way, construct a list of tokens, e.g. [5,0,2] => [0 0 0 0 0 2 2]
                    # token_list_list = [node_j for node_j in self.Nodes for counter_token in range(forward_token[msg_item][node_j])]
                    next_hop_token_pos = random.randrange(len(token_list[msg_item][msg_stage]))
                    #next_hop = token_list[next_hop_token_pos]
                    next_hop = token_list[msg_item][msg_stage].pop(next_hop_token_pos)
                    #if node == 0 and msg_item == 0:
                        #print('msg_item = '+str(msg_item))
                    #    print('token_list[msg_item] = '+str(token_list[msg_item]))
                    #    print('next_hop = '+str(next_hop)+', type = '+str(type(next_hop)))

                    forward_token_num[msg_item][msg_stage][next_hop] -= 1                    

                # send request msg to next hop
                if next_hop == 0:
                    if msg_stage[1] == 0:
                        #print('forward_token_num[msg_item][msg_stage] = '+str(forward_token_num[msg_item][msg_stage]))
                        #print('phi = '+str(self.Nodes[node]['RouteVar'][msg_item][msg_stage]))
                        #print('Warning: puting stage 0 request to computation')
                        pass
                    yield self.Nodes[node]['ComputeRequestPipe'].put(msg) 
                elif next_hop-1 in self.Nodes[node]['Neighbor']:
                    msg.Sender = node
                    yield self.Edges[node][next_hop-1]['LinkPipe'].put(msg)     
                else:
                    print('ERROR: Routed to non-link.')
                    exit()

    def compute_request_handling_process(self,node):
        """ Process that acts handles computation request that node decides to perform locally
            Make records for these computation requests, convert to the stage-1 request, and put into nodes request queue."""
        while True:
            msg = yield self.Nodes[node]['ComputeRequestPipe'].get()
            self.statistics['ComputeRequestRecord'][node] += 1
            msg_requester = msg.Source
            msg_item = msg.Item
            msg_stage = msg.Stage
            msg_id = msg.ID
            msg_sender = msg.Sender
            comp_instance = (msg_requester,msg_item,(msg_stage[0],msg_stage[1]-1),msg_id)

            # make record of where does this msg come from, used to route corr. response
            if comp_instance not in self.Nodes[node]['ComputeRequestRecord']:                        # if this msg is generated by this node, record the send time
                
                self.Nodes[node]['ComputeRequestRecord'][comp_instance] = self.env.now
            else:
                print('ERROR: Time:' +str(self.env.now)+'. Record already exist. Node:'+str(node)+', request instance:'+str(comp_instance)+', record:'+str(self.Nodes[node]['RequestRecord'][comp_instance]))
                exit()
                pass
            # convert the msg to stage-1, and put into node's request input queue
            if msg_stage[1] == 0:
                print('ERROR: pending computation request for stage 0')
                print('msg_instance = '+str(comp_instance))
                exit()
            else:
                msg.Stage = (msg_stage[0],msg_stage[1]-1)
                #print('generating pending comp request '+str(comp_instance))
                yield self.Nodes[node]['RequestPipe'].put(msg) 

    def response_handling_process(self,node):
        """ Process that acts as the response router for a node.
            Extract response messeages from the response input queue.
            If the response matches a exogenous request record, discard the msg and fulfill the request.
            If matches a pending computation request record, put into computation pipe.
            If matches a by-pass request record, forward the response in the reverse direction. """
        while True:
            msg = yield self.Nodes[node]['ResponsePipe'].get()
            self.statistics['HandledResponseMessages'] += 1
            msg_requester = msg.Source
            msg_item = msg.Item
            msg_stage = msg.Stage
            msg_id = msg.ID
            msg_sender = msg.Sender
            msg_payload = msg.Payload
            msg_instance = (msg_requester,msg_item,msg_stage,msg_id)
            #print('Handling response msg : '+str(msg_instance))
            msg.Payload = (msg_payload[0]+1, msg_payload[1])            # add hop number by 1
            self.statistics['CacheMissCost'][node][msg_item][msg_stage] += msg.Payload[1]    # every response with this neq 0 implies a cache miss, count the cache miss cost
            #print('CacheMissCost = '+str(self.statistics['CacheMissCost'][node]))

            # Check if this node is the request source, discard the response and update states
            if msg_instance in self.Nodes[node]['ExoRequestRecord']: 
                #print('Request satisfied: msg '+str(msg_instance)+' at node '+str(node))
                self.statistics['RequestSatisfied'] += 1
                t_send = self.Nodes[node]['RequestRecord'].pop(msg_instance)
                request_delay = self.env.now - t_send
                if request_delay < self.statistics['RequestDelay']['Min']:
                    self.statistics['RequestDelay']['Min'] = request_delay
                if request_delay > self.statistics['RequestDelay']['Max']:
                    self.statistics['RequestDelay']['Max'] = request_delay 
                avg_delay = self.statistics['RequestDelay']['Average']
                self.statistics['RequestDelay']['Average'] += (request_delay - avg_delay) / self.statistics['RequestSatisfied']
                #print('Request Finished : '+str(msg_instance))
            
            # if matches a previous pending computation request, put msg in to computation pipe
            elif msg_instance in self.Nodes[node]['ComputeRequestRecord']: 
                #print('resolving pending computation request '+str(msg_instance))
                self.Nodes[node]['ComputeRequestRecord'].pop(msg_instance)
                msg.Sender = node
                yield self.Nodes[node]['ComputePipe'].put(msg)   

            # if this node is not the requester or a pending computation, check the record
            elif msg_instance in self.Nodes[node]['RequestRecord']:                
                next_hop = self.Nodes[node]['RequestRecord'][msg_instance]
                self.Nodes[node]['RequestRecord'].pop(msg_instance)
                msg.Sender = node
                #print('node '+str(node)+' next-hop '+str(next_hop)+', is edge = '+str((node,next_hop) in self.EdgesEnumerator))
                yield self.Edges[node][next_hop]['LinkPipe'].put(msg)   

            else:
                print('ERROR: Record not found.')
                exit()

    def compute_process(self,node):
        """Process that performs the computation.
        Computation happens when pending computation requests retrives corr. data or intermediate result packet.
        Extract msg from computation pipe. Implement delay and promote stage, put the new msg to response pipe."""
        while True:
            msg = yield self.Nodes[node]['ComputePipe'].get()
            msg_requester = msg.Source
            msg_item = msg.Item
            msg_stage = msg.Stage
            msg_id = msg.ID
            msg_sender = msg.Sender
            msg_payload = msg.Payload
            msg_instance = (msg_requester,msg_item,msg_stage,msg_id)

            #print('Performing computation on msg '+str(msg_instance))
            self.statistics['ComputePerformed'][node] += 1

            # note: here should record computation workload for monitoring
            self.Nodes[node]['WorkloadCount'] += self.CompWorkload[msg_stage]

            # promote stage
            stage_new = (msg_stage[0],msg_stage[1]+1)
            if  stage_new[1] > self.AppChain_Len[msg_stage[0]]:
                print('ERROR: promote exceeding app len')
                exit()
            msg.Stage = stage_new
            msg.Size = self.PacketSize[msg_item][stage_new]
            yield self.Nodes[node]['ResponsePipe'].put(msg)   

            # here should apply delay


    def link_process(self,e):
        """Process that acts as the transmission link.
        Extract msg from link pipe. Implement delay according to config file. Send msgs to corr. router queue at receiver"""
        MargCost = self.LinkMargCal(e,0.0)  # margianl cost at F=0, used to calculate response aggregate cost
        #print('MargCost at link '+str(e)+' = '+str(MargCost))
        while True:
            msg = yield self.Edges[e[0]][e[1]]['LinkPipe'].get()
            self.Edges[e[0]][e[1]]['FlowCount'] += msg.Size

            if msg.MsgType == 'Request':
                if self.SimConfig.RequestDelayType == 'Deterministic':  # Deterministic request delay
                    delay = self.SimConfig.RequestDelay
                elif self.SimConfig.RequestDelayType == 'Exponential':  # Exponential request delay
                    if self.SimConfig.RequestDelay == 0.0:
                        delay = 0.0
                    else:
                        delay = random.expovariate(1.0 / self.SimConfig.RequestDelay)
                else:
                    print('ERROR: Undefined request delay type')
                    exit()
                yield self.env.timeout(delay)
                self.Nodes[e[1]]['RequestPipe'].put(msg)

            elif msg.MsgType == 'Response':
                if self.SimConfig.ResponseDelayType == 'Deterministic':  # Deterministic respond delay
                    delay = self.SimConfig.ResponseDelay
                elif self.SimConfig.ResponseDelayType == 'Exponential':  # Exponential request delay
                    if self.SimConfig.ResponseDelay == 0.0:
                        delay = 0.0
                    else:
                        delay = random.expovariate(1.0 / self.SimConfig.ResponseDelay)
                else:
                    print('ERROR: Undefined response delay type')
                    exit()
                old_payload = msg.Payload
                msg.Payload = (old_payload[0], old_payload[1] + MargCost)
                #print('response payload = '+str(msg.Payload))
                yield self.env.timeout(delay)
                self.Nodes[e[1]]['ResponsePipe'].put(msg)
            
            elif msg.MsgType == 'Control':
                # if ctrl msg, add D' to payload. Note this is from i to j, so j is the upstream node, and calculating pT/pr_j needs D'ij(Fij)
                Flow_e = self.Edges[e[0]][e[1]]['CurrentFlow']
                LinkMarg = self.LinkMargCal(e,Flow_e)
                payload_old = msg.Payload
                #if payload_old[0] >= 100.0:
                    #print('e = '+str(e)+', payload = '+str(payload_old))
                payload_new = (payload_old[0],LinkMarg,payload_old[2])
                if msg.Payload[1] is None:
                    msg.Payload = payload_new
                    pass
                else:
                    print("ERROR: Non-empty ctrl msg at edge " + str(e) + ", LinkMarg = " + str(LinkMarg) + ", msg = "+str(msg))
                    exit()
                delay = self.SimConfig.ControlDelay
                yield self.env.timeout(delay)
                self.Nodes[e[1]]['ControlPipe'][msg.Item].put(msg)

            else:
                print('ERROR: Undefined message type.')
                exit()

    def RRCache_random_shuffle_process(self,node):
        """ Shuffles RR cache every slot. """
        while True:
            yield self.env.timeout(self.SimConfig.T_slot)
            self.Nodes[node]['Cache'].refresh()

    def optCache_random_shuffle_process(self,node):
        """ Process that randomly determines cached item according to continuous cache variables y_im(k).
        This process is only activited when adopts optimized caches ('GP' or 'LMIN').
        This process will round the cache size of each cache type to integer, and randomly decide cached items in each type according to y_im(k) while keep the cache size to be steady."""
        #yield self.env.timeout(1.0)
        while True:
            yield self.env.timeout(self.SimConfig.T_slot)
            y_ori = self.Nodes[node]['CacheVar']
            #x = MultiTypeRoundingAlgo(y_ori)
            x = ElasticRounding(y_ori)
            #print('Node '+str(node)+', y= '+str(y_ori)+ ', update = '+str(x))
            self.Nodes[node]['Cache'].update(x)
            #if node == 41:
                #print('Cache var at 41 = '+str(y_ori))
                #print(' x_41 = '+str(x))
                #print('Content at node 41: '+str(self.Nodes[node]['Cache']._content))

    def optCache_mandate_shuffle(self, node):
        """ Mandatorily update cache according to cache var """
        y_ori = self.Nodes[node]['CacheVar']
        x = ElasticRounding(y_ori)
        #print('Node '+str(node)+', y= '+str(y_ori)+ ', update = '+str(x))
        self.Nodes[node]['Cache'].update(x)

    def control_generating_process(self,node,item):
        """ Only used for GP and distributed version of GCFW. Generates control msg to calculate pT/pt at the beginning of an update slot, 
        from the des server or cached item/result with y=1. """
        InpuMargInit = 0.0
        control_id = {}
        IsSendingCtrl = 0
        for stage in self.Stages:
            control_id[stage] = 0
        yield self.env.timeout(self.SimConfig.T_slot * (self.SimConfig.L_update -1))    # the first update slot is at (L-1)*T_slot, and then every each L*T_slot
        while True:
            for stage in self.Stages:
                # only generate control msg if is the designated server, or sum of y variables is 1.
                if stage == (0,0):   # for stage 0, we only broadcast (0,0) to represent the raw data
                    IsSendingCtrl = (node in self.DesServers[item]) or (self.Nodes[node]['CacheVar'][item][stage] >= 1.0 - eps )
                else:               # for other stages
                    IsSendingCtrl = (self.Nodes[node]['CacheVar'][item][stage] >= 1.0 - eps)

                if IsSendingCtrl and self.env.now <= self.SimConfig.T_sim:
                    Payload_msg = (InpuMargInit,None,False)     # initialzed to (pTpt, None(will be assigned L*D'ji(Fji), marker of 'improper')
                    Ctrlmsg = Message(MsgType='Control', ID=control_id[stage], Source=node,Item=item,Stage=stage,Payload=Payload_msg, Size=self.SimConfig.ControlSize) # MsgType,ID,Source,Item,Stage,Payload,Size
                    yield self.Nodes[node]['ControlPipe'][item].put(Ctrlmsg)
                    #print("Control msg initialied: node "+str(node)+". item "+str(item)+". stage "+str(stage))
                        #self.statistics['GeneratedControlMessages'] += 1
                control_id[stage] += 1
                IsSendingCtrl = 0
                # reset sum
                self.Nodes[node]['trafficMarg'][item][stage] = 0.0
            yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)

    def control_handling_process(self,node,item):
        """ Only used for GP and distributed version of GCFW. Takes msg from control msg input pipe hand handles. 
            This process also: (1) records the in-going  L*D'ji(Fji) + pTpt_j, 
            (2) calculates node input marginal pT/pt_i, 
            (3) keep track of 'imporper links'. """

        DownstreamCounter = {}      # marks if received ctrl msg from downstream nodes. {stage: control_id: [list of received node]}
        for stage in self.Stages:
            DownstreamCounter[stage] = {}
        while True:
            msg = yield self.Nodes[node]['ControlPipe'][item].get()
            msg_id = msg.ID
            msg_sender = msg.Sender
            msg_stage = msg.Stage
            DownstreamList = [ node_j for node_j,phi in enumerate(self.Nodes[node]['RouteVar'][item][msg_stage]) if phi >= eps ]    # note: here 0 stands for computing

            IsSendingCtrl = False
            if msg_sender == None:          # if this node is the des server or y=1
                IsSendingCtrl = True
            #elif msg_sender not in DownstreamList:      # discard ctrl msg from not downstream nodes
            #    IsSendingCtrl = False
            else:
                # record the msg
                MargRec = (msg.Payload[0], msg.Payload[1], msg.Payload[2])       # record pTpt_j, NaN and NaN
                #if MargRec[0] >= 10.0:
                    #print('Node '+str(node)+', item '+str(item)+', MargRec = '+str(MargRec))
                #print('MargRec = '+str(MargRec))
                
                
                # if sender is self, it represents computation 
                if msg_sender is node:
                    if (msg_stage[1] == 0) or ( 0 not in DownstreamList):
                        print('ERROR: receving computation controi form non-computing node')
                        exit()
                    else:
                        if msg_id not in DownstreamCounter[msg_stage]:
                            DownstreamCounter[msg_stage][msg_id] = [0]
                        elif 0 in DownstreamCounter[msg_stage][msg_id]:
                            print('ERROR: Repeated control message.')
                            exit()
                        else:
                            DownstreamCounter[msg_stage][msg_id].append(0)

                    # calculate the second term: Wimk *Ci'(Gi)
                    G_curr = self.Nodes[node]['CurrentWorkload']
                    CompMarg = self.CompMargCal(node,G_curr)
                    WeithedCompMarge = CompMarg * self.CompWorkload[msg_stage]
                    delay = self.SimConfig.ControlDelay
                    yield self.env.timeout(delay)

                    MargSum = MargRec[0] + WeithedCompMarge
                    self.Nodes[node]['MargRecord'][item][msg_stage][0] = (MargRec[0], WeithedCompMarge)
                    self.Nodes[node]['trafficMarg'][item][msg_stage] += self.Nodes[node]['RouteVar'][item][msg_stage][0] * MargSum

                    if sorted(DownstreamCounter[msg_stage][msg_id]) == sorted(DownstreamList):
                        IsSendingCtrl = True
                        pass

                # if sender is downstream node, record the sender in DownstreamCounter
                elif (msg_sender+1) in DownstreamList:
                    if msg_id not in DownstreamCounter[msg_stage]:
                        DownstreamCounter[msg_stage][msg_id] = [msg_sender+1]   # here +1 counts for 0 as computation
                    elif (msg_sender+1) in DownstreamCounter[msg_stage][msg_id]:
                        print('ERROR: Repeated control message.')
                        exit()
                    else:
                        DownstreamCounter[msg_stage][msg_id].append(msg_sender+1)

                    # add up to local input marginal: for stage (a,t), is the weighted sum of downstreams j: phi_ij* (L(a,t)*D'ji + pTpt_j), and the computation from stage (a,k-1): phi_i0* (W(iat)*C'i + pTpt_i_(a,k-1))
                    #if (not isinf(self.Nodes[node]['InputMarg'][item])) and isinf(MargSum):
                    #    print('Node '+str(node)+', item '+str(item)+', old InputMarg = '+str(self.Nodes[node]['InputMarg'][item])+', MargRec = '+str(MargRec))

                    # calculate the second term: Lak *Dji'(Fji)
                    e = (msg_sender,node)
                    F_curr = self.Edges[e[0]][e[1]]['CurrentFlow']
                    LinkMarg = self.LinkMargCal(e,F_curr)
                    WeithedLinkMarge = LinkMarg * self.PacketSize[item][msg_stage]
                    delay = self.SimConfig.ControlDelay
                    yield self.env.timeout(delay)

                    MargSum = MargRec[0] + WeithedLinkMarge
                    self.Nodes[node]['MargRecord'][item][msg_stage][msg_sender+1] = MargRec
                    self.Nodes[node]['trafficMarg'][item][msg_stage] += self.Nodes[node]['RouteVar'][item][msg_stage][msg_sender+1] * MargSum

                    # update node's improper marker
                    if msg.Payload[2] is True:
                        self.Nodes[node]['Improper'][item] = True

                    # if collected msg from all downstream nodes, send upstream
                    # Note: should only send upstream once, so activate only when receiving the last downstream ctrl msg
                    if sorted(DownstreamCounter[msg_stage][msg_id]) == sorted(DownstreamList):
                        IsSendingCtrl = True
                        pass
            
            # if downstream complete, further send ctrl msg to all its neighbors and local computation for (a,t+1)
            # Note: here, each neighbor should be assigned distinct object, multiple Message object has to be created
            if IsSendingCtrl:
                payload_new = (self.Nodes[node]['trafficMarg'][item][msg_stage], None, False) # here, improper is not implemented
                
                #  if current stage is not finial stage, broadcast should include computation
                #  here, computation msg is directly handled
                if msg_stage[1] < self.AppChain_Len[msg_stage[0]]:
                    # if is the raw data, promote to all stage 1
                    if msg_stage[1] == 0:
                        for app in range(self.number_of_applications):
                            if self.AppChain_Len[app] >= 1:
                                stage_promote = (app,1)
                                msg_new = Message(MsgType='Control', ID=msg_id, Source=node,Item=item,Stage=stage_promote, Payload=payload_new, Size=self.SimConfig.ControlSize)
                                msg_new.Sender = node
                                self.statistics['GeneratedControlMessages'] += 1
                                yield self.Nodes[node]['ControlPipe'][item].put(msg_new)
                                pass
                    
                    # if not raw data, only promote to self stage
                    else:
                        stage_promote = (msg_stage[0],msg_stage[1]+1)
                        msg_new = Message(MsgType='Control', ID=msg_id, Source=node,Item=item,Stage=stage_promote, Payload=payload_new, Size=self.SimConfig.ControlSize)
                        msg_new.Sender = node
                        self.statistics['GeneratedControlMessages'] += 1
                        yield self.Nodes[node]['ControlPipe'][item].put(msg_new)
                        pass
                
                # broadcast to neighbor nodes
                for next_hop in self.Nodes[node]['Neighbor']:
                    msg_new = Message(MsgType='Control', ID=msg_id, Source=node,Item=item,Stage=msg_stage,Payload=payload_new, Size=self.SimConfig.ControlSize)
                    msg_new.Sender = node
                    #print("Sending ctrl from node "+str(node)+" for item "+str(item)+", time:"+str(self.env.now))
                    self.statistics['GeneratedControlMessages'] += 1
                    #print('Node '+str(node)+', item '+str(item)+', sending ctrl msg with payload = '+str(payload_new))
                    yield self.Nodes[next_hop]['ControlPipe'][item].put(msg_new)
                
                

    def block_node_process(self):
        """ Only used in GP. Process that updates the blocked node sets, at the beginning of each update slot."""
        yield self.env.timeout(self.SimConfig.T_slot * (self.SimConfig.L_update -1))    # the first update slot is at (L-1)*T_slot, and then every each L*T_slot
        while True:
            # init
            for i in self.Nodes:
                for k in self.Items:
                    for s in self.Stages:
                        self.Nodes[i]['BlockNodes'][k][s] = [ j for j in self.Nodes if j not in self.Nodes[i]['Neighbor'] ]

            if self.AlgoConfig.AllowLoop is False:
                # if loops are not allowed, practice blocked node sets: topological sort
                for k in self.Items:
                    for s in self.Stages:
                        # first construct the DAG for content k and stage s
                        DAG_k = nx.DiGraph()
                        DAG_k.add_nodes_from(self.Nodes.keys())
                        for i in self.Nodes:
                            for j in self.Nodes[i]['Neighbor']:
                                if self.Nodes[i]['RouteVar'][k][s][j+1] >= eps: # here +1 counts for computation
                                    DAG_k.add_edge(i,j)
                        # then run topological sort
                        #print(DAG_k)
                        if not nx.is_directed_acyclic_graph(DAG_k):
                            print('ERROR: Not starting with DAG.')
                            exit()
                        order_k = list(nx.topological_sort(DAG_k))
                        #print('k='+str(k)+', source = '+str(self.DesServers[k][0])+', order= '+str(order_k))
                        # then update blocked nodes
                        for i_pos in range(self.number_of_nodes):
                            i = order_k[i_pos]
                            for j_pos in range(i_pos):      # blocked nodes: order(0) to order(i_pos -1), note range(0) = [], range(1) = [0]
                                j = order_k[j_pos]
                                if j not in self.Nodes[i]['BlockNodes'][k][s]:
                                    self.Nodes[i]['BlockNodes'][k][s].append(j)
                            #print('i= '+str(i)+', blocked nodes:'+str(self.Nodes[i]['BlockNodes'][k]))
                        #print(self.Nodes[i]['BlockNodes'][k])
            yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)

    def update_process_GP(self,node,item,stage):
        """ Routing and caching variable updating process, for GP 
            Update [node]['RouteVar'][item][stage] and [node]['CacheVar'][item][stage] at the end of each update slot.
            The update depend on (D'ji(Fji)+ pTpr_j) and blocked node set. """
        # update at the end of each update slot
        Period_len = self.SimConfig.T_slot * self.SimConfig.L_update
        yield self.env.timeout(0.1)
        while True:
            yield self.env.timeout(Period_len)
            #print('Time: '+str(self.env.now)+', updating (GP) ...')

            # for raw data and designated server, phi = y = 0
            if stage[1] == 0 and node in self.DesServers[item]:
                self.Nodes[node]['RouteVar'][item][stage] = np.zeros(self.number_of_nodes +1)
                self.Nodes[node]['CacheVar'][item][stage] = 0.0
                continue

            # (use the blocked nodes set by block_node_process)
            unblock_nodes = [j for j in self.Nodes[node]['Neighbor'] if j not in self.Nodes[node]['BlockNodes'][item][stage]]
            #print('node = '+str(node) +', item = '+str(item)+', unblock_nodes = '+str(unblock_nodes))

            # necessary stats for calculationg Delta
            # calculate tik and reset counter
            if stage == (0,0):  # for raw data, only consider stage (0,0). The updated phi will be copied for all applications
                traffic = self.Nodes[node]['RequestCount'][item][stage] / Period_len
                self.Nodes[node]['RequestCount'][item][stage] = 0.0
            elif stage[1]>0:    # if not raw data, proceed
                traffic = self.Nodes[node]['RequestCount'][item][stage] / Period_len
                self.Nodes[node]['RequestCount'][item][stage] = 0.0
            else:               # if  (x,0), jump
                continue
            #print('t_ik = '+str(tik))

            # calculate pTpr_j + LD'ji(F_ji) for j in N_i\B_i(k)
            # Note: out_marg_routing only containts unblocked nodes, so the index is not the same as node ID.
            out_marg_routing = np.array([ self.Nodes[node]['MargRecord'][item][stage][j+1][0] + self.Nodes[node]['MargRecord'][item][stage][j+1][1] \
                for j in unblock_nodes])  

            # if is not the final stage, calculate comp marg t*(WC' + pTpt_(a,t-1)) and put at 0. For final stage, put inf
            if stage[1] < self.AppChain_Len[stage[0]]:
                compMarg = self.Nodes[node]['MargRecord'][item][stage][0][0] + self.Nodes[node]['MargRecord'][item][stage][0][1]
                out_marg_routing.insert(0,compMarg)
            else:
                out_marg_routing.insert(0,np.Inf)
            #print('out_marg_routing = '+str(out_marg_routing))   

            # calculate LB'i(Yi) / ti(k) 
            cachesize_theo = 0.0
            for k in self.Items:
                for s in self.Stages:
                    if s == (0,0) or s[1]>0:    # for raw data, only count the cache for (0,0).
                        cachesize_theo += self.Nodes[node]['CacheVar'][k][s]
            #print('self.CacheTypes = '+str(self.CacheTypes))
            #print(' cachesize_theo = '+str(cachesize_theo))
            #print(' CacheCostType = '+str(self.CacheCostType))
            #print(' Cache cost = '+str(self.CacheMargCal(node,0,cachesize_theo[0])))
            out_marg_caching = self.CacheMargCal(node,cachesize_theo) / traffic if traffic > eps else np.Inf
            #print('out_marg_caching = '+str(out_marg_caching))
            #min_caching_pos = out_marg_caching.argmin()
            
            # find the minimum marg
            if len(out_marg_routing) != 0:
                # if there exists at least one out-avaialbe
                min_routing_pos = out_marg_routing.argmin()
                if out_marg_routing[min_routing_pos] <= out_marg_caching:
                    delta_min = out_marg_routing[min_routing_pos]
                    min_set_routing = [min_routing_pos]
                    min_set_caching = []
                else:
                    delta_min = out_marg_caching
                    min_set_routing = []
                    min_set_caching = [0]
            else:
                # if all out-neighbors are blocked
                delta_min = out_marg_caching

            #min_set_routing = [unblock_nodes[j_pos] for j_pos in range(len(unblock_nodes)) if out_marg_routing[j_pos] == delta_min ]
            #min_set_caching = [m for m in self.CacheTypes if out_marg_caching[m] == delta_min ]
            P_min = len(min_set_routing)
            Q_min = len(min_set_caching)
            #print('min_set_routing = '+str(min_set_routing)+', min_set_caching = '+str(min_set_caching) + ', N_min = '+str(N_min))
            # [e_ij] and [e_im]
            e_routing = out_marg_routing.copy() - delta_min if len(out_marg_routing) >= 1 else []
            e_caching = out_marg_caching.copy() - delta_min if len(out_marg_caching) >= 1 else []
            #print('delta_min = '+str(delta_min))
            #print('e_routing = '+str(e_routing))
            #print('e_caching = '+str(e_caching))

            # normalize e to max-1
            #max_e = max( e_routing[np.isfinite(e_routing)] ) if len(e_routing[np.isfinite(e_routing)]) >= 1 else 0.0 #, max(e_caching) )
            max_e = 0.0
            for e_ij in e_routing:
                if isfinite(e_ij) and  e_ij > max_e:
                    max_e = e_ij
            if max_e > eps:
                e_routing = e_routing / max_e
                e_caching = e_caching / max_e
                pass

            # Calculate Delta
            decreased_sum_routing = 0.0         # sum of decrease amount of Delta \phi or y, will be equaly assigned to min_delta
            decreased_sum_caching = 0.0
            # first calculate sum_phi
            Delta_phi = np.zeros(self.number_of_nodes +1)
            for j in self.Nodes:
                #print('j = '+str(j)+', phi = '+str(self.Nodes[node]['RouteVar'][item][j]))
                # for self and blocked nodes, decrease routing var to 0 (should already be 0, just strip small values)
                if j is node:
                    Delta_phi[j] = -1.0 * self.Nodes[node]['RouteVar'][item][j]
                    decreased_sum_routing += -1.0 * Delta_phi[j]
                elif j not in unblock_nodes:
                    Delta_phi[j] = -1.0 * self.Nodes[node]['RouteVar'][item][j]
                    decreased_sum_routing += -1.0 * Delta_phi[j]
                # for unblocked nodes NOT corr. to delta min, decrease according to stepsize
                elif j not in min_set_routing:
                    j_pos = unblock_nodes.index(j)
                    # step weighted by tik
                    #step_ori = (self.AlgoConfig.StepsizeGP_phi * e_routing[j_pos] / tik) if tik > eps else np.Inf
                    # step not weigted
                    step_ori = self.AlgoConfig.StepsizeGP_phi * e_routing[j_pos]
                    step = np.minimum( self.Nodes[node]['RouteVar'][item][j], step_ori)                                 # step = min(phi_ij,  alpha / tik * e_ij(k))
                    Delta_phi[j] = -1.0 * step
                    decreased_sum_routing += -1.0 * Delta_phi[j]
            # then sum_y
            Delta_y = np.zeros(self.M)
            for m in self.CacheTypes:
                if m not in min_set_caching:
                    #step_ori = (self.AlgoConfig.StepsizeGP_y * e_caching[m] / tik) if tik > eps else np.Inf
                    step_ori = self.AlgoConfig.StepsizeGP_y * e_caching[m] 
                    step = np.minimum( self.Nodes[node]['CacheVar'][item][m] , step_ori)                                # step = min(y_im,  beta / tik * e_im(k))
                    #if step > 0.0:  print('StepsizeGP_y = '+str(self.AlgoConfig.StepsizeGP_y) + ', step = '+str(step))
                    Delta_y[m] = -1.0 * step
                    decreased_sum_caching += -1.0 * Delta_y[m]
            # then equally assign to mins
            #print('decreased_sum = '+str(decreased_sum))
            routing_step_frac = self.AlgoConfig.StepsizeGP_phi /(self.AlgoConfig.StepsizeGP_phi + self.AlgoConfig.StepsizeGP_y)
            caching_step_frac = self.AlgoConfig.StepsizeGP_y /(self.AlgoConfig.StepsizeGP_phi + self.AlgoConfig.StepsizeGP_y)
            #print('caching_step_frac = '+str(caching_step_frac))

            # find the opt routing and opt caching separately
            #opt_routing_j = [unblock_nodes[j_pos] for j_pos in range(len(unblock_nodes)) if np.abs(out_marg_routing[j_pos] - out_marg_routing.min()) <= eps ]
            opt_routing_j = [unblock_nodes[j_pos] for j_pos in range(len(unblock_nodes)) if out_marg_routing[j_pos] == out_marg_routing.min() ]
            #opt_caching_m = [m for m in self.CacheTypes if np.abs(out_marg_caching[m] - out_marg_caching.min()) <= eps]
            opt_caching_m = [m for m in self.CacheTypes if out_marg_caching[m] == out_marg_caching.min()]

            # set the thorttle rate of routing-caching exchanging. 
            # If Thrt == 0, the routing sum will only be distributed to min-routing vars, and caching sum only to min-caching vars
            # If Thrt == 1, the routing sum and caching sum will be totally merged and distributed to the min-delta vars regardless of routing or caching
            # Specifically, apart from 'decreased_sum_routing' and 'decreased_sum_caching', a third sum-pool 'decreased_sum_mix' is maintained. 
            # 'decreased_sum_mix' absorbs the fraction of 'Thrt' of both 'decreased_sum_routing' and 'decreased_sum_caching', and equally distributed to all min-delta vars (routing and caching).
            # Meanwhile, the remained 'decreased_sum_routing' is equally distributed to all min-routing vars, and similar as 'decreased_sum_caching'.
            # The value of Thrt is either given explicitly in config file, or set to 'Auto' where Thrt = min(stepsize_routing, stepsize_caching) / max(stepsize_routing, stepsize_caching)
            if self.AlgoConfig.Thrt == 'Auto':
                throttle_rate = min(self.AlgoConfig.StepsizeGP_phi, self.AlgoConfig.StepsizeGP_y) / ( eps + max(self.AlgoConfig.StepsizeGP_phi, self.AlgoConfig.StepsizeGP_y) )
            else:
                throttle_rate = self.AlgoConfig.Thrt
            decreased_sum_mix = throttle_rate * (decreased_sum_routing + decreased_sum_caching)
            decreased_sum_routing -= decreased_sum_routing * throttle_rate
            decreased_sum_caching -= decreased_sum_caching * throttle_rate
            
            # first disrtibute routiong-pool and caching-pool separately
            #for j in min_set_routing:
            for j in opt_routing_j:
                if Delta_phi[j] != 0.0:
                    pass
                    #print('Non-zero in min routing set')
                #Delta_phi[j] = decreased_sum * routing_step_frac / P_min
                #Delta_phi[j] = decreased_sum / (P_min + Q_min)
                #Delta_phi[j] += decreased_sum * routing_step_frac / len(opt_routing_j)
                Delta_phi[j] += decreased_sum_routing / len(opt_routing_j)

            #for m in min_set_caching:
            for m in opt_caching_m:
                #Delta_y[m] = decreased_sum * caching_step_frac / Q_min
                #Delta_y[m] = decreased_sum / (P_min + Q_min)
                #Delta_y[m] += decreased_sum *caching_step_frac / len(opt_caching_m)
                Delta_y[m] += decreased_sum_caching  / len(opt_caching_m)
            
            # the distribute mix-pool equally to all min-delta vars
            for j in min_set_routing:
                Delta_phi[j] += decreased_sum_mix / (P_min + Q_min)
            for m in min_set_caching:
                Delta_y[m] += decreased_sum_mix  / (P_min + Q_min)

            # update variables
            for j in self.Nodes:
                self.Nodes[node]['RouteVar'][item][j] += Delta_phi[j]
            for m in self.CacheTypes:
                self.Nodes[node]['CacheVar'][item][m] += Delta_y[m]
                        
            # test if sum up to 1
            sum = 0.0
            for j in self.Nodes:
                sum += self.Nodes[node]['RouteVar'][item][j]
            for m in self.CacheTypes:
                sum += self.Nodes[node]['CacheVar'][item][m]
            if np.abs(sum - 1.0) > eps:
                print('Warning: Time:'+str(self.env.now)+', node '+str(node)+', item '+str(item)+', variables summing to '+str(sum))
                for j in self.Nodes:
                    self.Nodes[node]['RouteVar'][item][j] *= 1.0/sum
                for m in self.CacheTypes:
                    self.Nodes[node]['CacheVar'][item][m] *= 1.0/sum
                #print(' Sum of Delta_phi = '+str(np.sum(Delta_phi))+', sum of Delta_y = '+str(np.sum(Delta_y))+', decreased_sum = '+str(decreased_sum))
                print('min_set_routing = '+str(min_set_routing)+', min_set_caching = '+str(min_set_caching))
                pass

            #if self.Nodes[node]['CacheVar'][item][0] > eps:
            #if cachesize_theo[0] > 1.0:
            #if node == 0 and item == 0:
            #if  np.isinf(delta_min):
            #if False:
                pass
                print(LineBreak)
                print('Node '+str(node)+', item '+str(item))
                print('Neighbors = '+str(self.Nodes[node]['Neighbor']))
                print('t_ik = '+str(tik))
                #print('Node '+str(node)+', item '+str(item)+': out_marg_routing + out_marg_caching = '+str(out_marg_routing) + ' ; '+ str(out_marg_caching) )
                print('self.Nodes[node][MargRecord][item] = '+str(self.Nodes[node]['MargRecord'][item]))
                print('out_marg_routing + out_marg_caching = '+str(out_marg_routing) + ' ; '+ str(out_marg_caching) )
                print('cachesize_theo = '+str(cachesize_theo))
                print('unblock_nodes = '+str(unblock_nodes))
                print('opt_routing_j = '+str(opt_routing_j)+', opt_caching_m = '+str(opt_caching_m))
                print('e_routing = '+str(e_routing))
                print('e_caching = '+str(e_caching))
                print('Delta_phi = '+str(Delta_phi))
                print('Delta_y = '+str(Delta_y))
                #print('Node '+str(node)+', item '+str(item)+': var_routing + var_caching = '+str(self.Nodes[node]['RouteVar'][item]) + ' ; '+ str(self.Nodes[node]['CacheVar'][item]) )
                print('var_routing + var_caching = '+str(self.Nodes[node]['RouteVar'][item]) + ' ; '+ str(self.Nodes[node]['CacheVar'][item]) )
                #print('F_03 = '+str(self.Edges[0][3]['CurrentFlow'])+', D_03 = '+str(self.LinkMargCal((0,3),self.Edges[0][3]['CurrentFlow'])))
            
            #for node in self.Nodes:
            #    self.optCache_mandate_shuffle(node)
                
    def topo_sort(G):
        """ topological sort on graph """

    def update_process_GCFW(self,N):
        """ Gradient Combined Frank-Wolfe (GCFW), a centralized algorithm for the fixed-routing special case.
        The algorithm do not change the routing path generated by the initialization (but the routing variable may change). 
        The algorithm solves a linear progamming every iteration, the objective direction is stated in the paper.
        N: number of iterations. """
        eps_GCFW = N**(-1/3)
        #eps_GCFW = 0.08

        # first, identify all paths p_vk for all v and all k
        paths = {}  # path[node][item] = [list of p_vk, the path from v to a server of k]
        for node in self.Nodes:
            paths[node] = {}
            for item in self.Items:
                paths[node][item] = []
        for v in self.Nodes:
            for k in self.Items:
                current_node = v
                while True:
                    paths[v][k].append(current_node)
                    if current_node in self.DesServers[k]:
                        break
                    else:
                        forward_var = self.Nodes[current_node]['RouteVar'][k]
                        max_var = max(forward_var)
                        if max_var < 0.5:
                            print('Warning: max_var = '+str(max_var))
                        next_node = forward_var.argmax()
                        current_node = next_node
        
        # update each time slot, according to the theoretcial cost
        best_theo_cost = np.inf
        best_hist_sol = {}
        for node in self.Nodes:
            best_hist_sol[node] = {}
            best_hist_sol[node]['RouteVar'] = {}
            best_hist_sol[node]['CacheVar'] = {}
            for item in self.Items:
                best_hist_sol[node]['RouteVar'][item] = []
                best_hist_sol[node]['CacheVar'][item] = []
        iter_num = 0

        while True:
            yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)
            iter_num += 1

            if iter_num == N+1:
                # if all N updates are finished, pick the variable that corresponds to the best theo performance
                for node in self.Nodes:
                    for item in self.Items:
                        self.Nodes[node]['RouteVar'][item] = best_hist_sol[node]['RouteVar'][item][:]   # copy the historical best solution
                        self.Nodes[node]['CacheVar'][item] = best_hist_sol[node]['CacheVar'][item][:]

            elif iter_num > N+1:
                # for any future slots, keep the variable unchanged
                continue

            else:
                # GCFW update
                # first calculate the theoretical costs (of last set of variables) and compare with historical best
                (TheoLinkCost,TheoCacheCost) = self.TheoreticalCostCal()
                if TheoLinkCost + TheoCacheCost < best_theo_cost:
                    for node in self.Nodes:
                        for item in self.Items:
                            best_hist_sol[node]['RouteVar'][item] = self.Nodes[node]['RouteVar'][item][:]
                            best_hist_sol[node]['CacheVar'][item] = self.Nodes[node]['CacheVar'][item][:]
    
                # solving the linear programming in frank-wolfe is simply equivalent to letting the (v,k) with positive derivative be 1, and others be 0.
                s_GCFW = {}
                s_sum = 0.0
                (TheoFlow, TheoTraffic) = self.TheoreticalFlowCal()
                for node_z in self.Nodes:
                    s_GCFW[node_z] = {}
                    for item in self.Items:
                        s_GCFW[node_z][item] = 0.0

                        # calculate the corr. derivative, first partial A / partial Y
                        pApY = 0.0
                        path = paths[node_z][item]

                        tzk = TheoTraffic[item][node_z]     # t_z(k)
                        prod = 1.0                        # the product of (1 - Y) from the downstream neighbor of z to current node on path
                        for path_pos in range(len(path)-1):      # if |path| = 1, no instance
                            node_i = path[path_pos]
                            node_j = path[path_pos +1]

                            # D'ji(Fji)
                            F_ji = TheoFlow[(node_j,node_i)]
                            D_marg = self.LinkMargCal((node_j,node_i),F_ji)

                            # use prod and D' to calculate pApY
                            pApY += D_marg * tzk * prod

                            # update prod with Y_j,k
                            Y_j = sum(self.Nodes[node_j]['CacheVar'][item])
                            prod *= (1 - Y_j) 
                        
                        # then partial B* / partial Y
                        if len(self.CacheTypes) == 1:
                            # for the special case of only one cache type
                            cachesize = sum(self.Nodes[node_z]['CacheVar'][item])
                            pBpY = self.CacheMargCal(node_z,0,cachesize)
                        else:
                            print('ERROR: Calculation of B* gradient for M > 1 is not yet implemented.')
                            exit()

                        # then if gradient G >0, the corr. s is set to 1, else 0
                        combined_grad = pApY - 2*pBpY
                        if combined_grad > 0:
                            s_GCFW[node_z][item] = 1.0
                            s_sum += 1.0
                        else:
                            s_GCFW[node_z][item] = 0.0
                print('GCFW: sum of s = '+str(s_sum))

                # update the variables according to s
                for node in self.Nodes:
                    for item in self.Items:

                        # first adjust the caching variables
                        if len(self.CacheTypes) == 1:
                            self.Nodes[node]['CacheVar'][item][0] = \
                                self.Nodes[node]['CacheVar'][item][0] * (1 - eps_GCFW**2) + s_GCFW[node][item] * eps_GCFW**2
                        else:
                            print('ERROR: update of caching variable for M > 1 is not yet implemented.')
                        
                        # then the routing variables, note that a fixed routing scheme is assumed
                        if node in self.DesServers[item]:
                            for j in self.Nodes:
                                self.Nodes[node]['RouteVar'][item][j] = 0
                        else:
                            next_node = paths[node][item][1]
                            for j in self.Nodes:
                                if j == next_node:
                                    self.Nodes[node]['RouteVar'][item][j] = 1.0 - sum(self.Nodes[node]['CacheVar'][item])
                                else:
                                    self.Nodes[node]['RouteVar'][item][j] = 0

    def update_process_ContinuousGreedy(self):
        """FAIL!!! Continuous greedy caching algorithm (cf. Kelly Cache Network)
        The step number is set to 10 to speed up.
        Gradient is evaluated using theoretical link flows"""
        N_CG = 10
        gamma_CG = 1.0 / N_CG

        # first, identify all paths p_vk for all v and all k
        paths = {}  # path[node][item] = [list of p_vk, the path from v to a server of k]
        for node in self.Nodes:
            paths[node] = {}
            for item in self.Items:
                paths[node][item] = []
        for v in self.Nodes:
            for k in self.Items:
                current_node = v
                while True:
                    paths[v][k].append(current_node)
                    if current_node in self.DesServers[k]:
                        break
                    else:
                        forward_var = self.Nodes[current_node]['RouteVar'][k]
                        max_var = max(forward_var)
                        if max_var < 0.5:
                            print('Warning: max_var = '+str(max_var))
                        next_node = forward_var.argmax()
                        current_node = next_node
        
        # update each time slot, according to the theoretcial cost
        while True:
            
            # update every period
            yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)

            y_CG = {}   # initial y0 = 0, no cache, at every period
            for node in self.Nodes:
                y_CG[node] = np.zeros(self.number_of_items)

            for iter_id in range(N_CG):
                # at each slot, calculate mk : the LP result
                # For each node, calculate the gradient of caching gain, and cache the most large c_v items
                totalCacheSize = 0.0
                (TheoFlow, TheoTraffic) = self.TheoreticalFlowCal() # theo cost at current slot
                mk_items = {}
                for node in self.Nodes:
                    gradient_items = np.zeros(self.number_of_items) # the gradient of caching gain w.r.t. y_vi
                    for item in self.Items:
                        pGpY = 0.0
                        path = paths[node][item]
                        tvk = TheoTraffic[item][node]     # t_v(k)
                        prod = 1.0                        # the product of (1 - Y) from the downstream neighbor of z to current node on path
                        for path_pos in range(len(path)-1):      # if |path| = 1, no instance
                            node_i = path[path_pos]
                            node_j = path[path_pos +1]

                            # D'ji(Fji)
                            F_ji = TheoFlow[(node_j,node_i)]
                            D_marg = self.LinkMargCal((node_j,node_i),F_ji)

                            # use prod and D' to calculate pApY
                            pGpY += D_marg * tvk * prod

                            # update prod with Y_j,k
                            Y_j = sum(self.Nodes[node_j]['CacheVar'][item])
                            prod *= (1 - Y_j) 
                        gradient_items[item] = pGpY
                    
                    # then pick the top items
                    #mk_node = np.zeros(self.number_of_items)
                    cacheSize_node = 0.0
                    for m in self.CacheTypes:
                        cacheSize_node += self.Nodes[node]['Cache'].size[m]
                    if cacheSize_node - np.round(cacheSize_node) > eps:
                        print('Warning: Cache size not integer in Continuous-Greedy.')
                    cacheSize_node = int(np.round(cacheSize_node))
                    # pick the items
                    totalCacheSize += cacheSize_node
                    mk_items[node] = []
                    if cacheSize_node != 0:
                        #gradient_items_sorted=list(gradient_items).sort()
                        SortedIndex = sorted(range(len(gradient_items)), key=lambda x: gradient_items[x], reverse=True)
                        for t in range(cacheSize_node):
                            mk_items[node].append(SortedIndex[t])                    

                # after get mk, update y_CG 
                for node in self.Nodes:
                    for item in self.Items:
                        if item in mk_items[node]:
                            y_CG[node][item] += gamma_CG
                                
            # After running a whole CG, update real variables
            # first adjust the caching variables
            total_cache_size = 0.0
            for node in self.Nodes:
                y_sum = 0.0
                for item in self.Items:
                    if len(self.CacheTypes) == 1:
                        if y_CG[node][item] >= 0 and y_CG[node][item] <= 1.0+eps:
                            #print('ERROR: wrong caching variable '+str(self.Nodes[node]['CacheVar'][item][0]))
                            self.Nodes[node]['CacheVar'][item][0] = min(y_CG[node][item],1.0)
                            y_sum += self.Nodes[node]['CacheVar'][item][0]
                        else:
                            print('ERROR: wrong caching variable y_CG'+str(y_CG[node][item]))
                    else:
                        print('ERROR: update of caching variable for M > 1 is not yet implemented.')
            # then routing variables, set to 1-yi
                    if node in self.DesServers[item]:
                        for j in self.Nodes:
                            self.Nodes[node]['RouteVar'][item][j] = 0
                    else:
                        next_node = paths[node][item][1]
                        for j in self.Nodes:
                            if j == next_node:
                                self.Nodes[node]['RouteVar'][item][j] = 1.0 - sum(self.Nodes[node]['CacheVar'][item])
                            else:
                                self.Nodes[node]['RouteVar'][item][j] = 0
                actual_cache_size = sum(self.Nodes[node]['Cache'].size)
                total_cache_size += actual_cache_size
                if abs(actual_cache_size - y_sum) >= eps:
                    print('Warning: non-compatable cache size. Actual size = '+str(actual_cache_size)+', sum_y = '+str(y_sum))
            # monitor the theoretical cost
            (linkCost_theo, CacheCost_theo) = self.TheoreticalCostCal()
            TotalCost_theo = linkCost_theo + CacheCost_theo                
            print('Updated Continuous-Greedy, Total theo cost = '+str(TotalCost_theo)+', total cache size = '+str(total_cache_size))
                
    def update_process_Greedy(self):
        """Greedy algorithm with 1/2 guarantee, cf. Kelly Cache Networks
        Using a fixed routing at initialization. At each period, run an update.
        In each update, start with x=0. iteratively check all valid e_{vk}, pick the largest gain, untill fill all cache space."""
        
        # first identify all paths, all node w.r.t. all items
        RouteVar = {}           # save the original routing variables
        NextNode = {}           # save the next hop nodes
        for node in self.Nodes:
            NextNode[node] = {}
            RouteVar[node] = {}
            for item in self.Items:
                if node in self.DesServers[item]:
                    NextNode[node][item] = []
                    RouteVar[node][item] = np.zeros(self.number_of_nodes)
                else:
                    NextNode[node][item] = []
                    RouteVar[node][item] = list(self.Nodes[node]['RouteVar'][item])
                    #routVarList = self.Nodes[node]['RouteVar'][item]
                    nexthop_num = 0
                    for node_j in self.Nodes:
                        if RouteVar[node][item][node_j] > eps:
                            nexthop_num += 1
                            NextNode[node][item].append(node_j)
                    if nexthop_num > 1:
                        print('Warning: Non-single path routing in Greedy')
        # then identify all edges that could be effected, all node w.r.t. all items
        #maxhop = 50
        Effect_edges = {}
        for node in self.Nodes:
            Effect_edges[node] = {}
            for item in self.Items:
                Effect_edges[node][item] = []
                # run a BFS to identify all effected links
                remaining_nodes = [node]
                while remaining_nodes:
                    exam_node = remaining_nodes[0]
                    for exam_node_j in NextNode[exam_node][item]:
                        #edge = (exam_node,exam_node_j)
                        edge = (exam_node_j,exam_node)          # note: here, should be (j,i) 
                        if edge in Effect_edges[node][item]:
                            print('ERROR: loop detected in Greedy.')
                        else:
                            Effect_edges[node][item].append(edge)
                            remaining_nodes.append(exam_node_j)
                    remaining_nodes.pop(0)
        #print('Effect_edges[0][0] = '+str(Effect_edges[0][0]))

        yield self.env.timeout(0.5)
        while True:
            # update every period
            yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)

            x_Greedy = {}
            cache_sizes = {}
            for node in self.Nodes:
                x_Greedy[node] = [0] * self.number_of_items
                cache_sizes[node] = sum(self.Nodes[node]['Cache'].size)
            total_cache_size = sum(cache_sizes.values())
            if abs(total_cache_size - np.round(total_cache_size)) > eps:
                print('Warning: Non-integer cache size in Greedy, will round')
            total_cache_size = int(np.round(total_cache_size))
            #print('cache_sizes = '+str(cache_sizes))

            phi_Greedy = CopyRoutVar(RouteVar)    # start with the initial routing state, should be no cache and loop free
            print('Greedy updateing... Total cache size = '+str(total_cache_size))
            for iter in range(total_cache_size):
                # in each iteration, compare cost for all e_{vi}, pick the largest
                # first record all f_ijk with current x_Greedy, and record all Dij
                # then for all valid e_vi, remove the f_ijk for all (i,j) effected
                # then the Delta_evi is the sum off all decrease value on these effected (i,j)
                #f_current = self.TheoreticalFlowCal_GivenX(x_Greedy)     # {edge ij: item k: f_ijk} 
                #F_current = {}                                           # {edge ij: F_ij}
                #for e in self.EdgesEnumerator:
                #    F_current[e] = sum(f_current[e].values())
                #print('sum F_current = '+str(sum(F_current.values())))
                #print('F_10,26 = '+str(F_current[(10,26)]))
                # identify all avaiable e_vi
                possible_inc = []       # available (node, item) to be newly cached
                for node in self.Nodes:
                    if sum(x_Greedy[node]) <= cache_sizes[node] - 1.0:
                        for item in self.Items:
                            if x_Greedy[node][item] == 0:
                                possible_inc.append((node,item))
                #print('possible_inc = '+str(possible_inc))
                Delta_evi = {}
                #BaseLinkCost = self.TheoreticalLinkCostCal_GivenPhi(phi_Greedy)
                BaseLineLinkflow = self.TheoreticalFlowCal_GivenPhi(phi_Greedy)
                #print('BaseLinkCost = '+str(BaseLinkCost))
                #phi_Greedy_temp = CopyRoutVar(phi_Greedy)
                for (node_inc,item_inc) in possible_inc:
                    #phi_Greedy_inc = copy.deepcopy(phi_Greedy)
                    #phi_Greedy_temp[node_inc][item_inc] = np.zeros(self.number_of_nodes)
                    #LinkCost_inc = self.TheoreticalLinkCostCal_GivenPhi(phi_Greedy_temp)
                    #Delta_evi[(node_inc,item_inc)] = BaseLinkCost - LinkCost_inc
                    Delta_evi[(node_inc,item_inc)] = 0.0
                    for e in Effect_edges[node_inc][item_inc]:
                        F_e = sum(BaseLineLinkflow[e].values())
                        delta = self.LinkCostCal(e,F_e) \
                            - self.LinkCostCal(e, F_e - BaseLineLinkflow[e][item_inc])
                        #print('F_current[e] = '+str(F_current[e])+', f_current[e][item_inc] = '+str(f_current[e][item_inc])+', delta = '+str(delta))
                        Delta_evi[(node_inc,item_inc)] += delta
                    #phi_Greedy_temp[node_inc][item_inc] = phi_Greedy[node_inc][item_inc].copy()
                (node_inc_max,item_inc_max) = max(Delta_evi, key = Delta_evi.get) 
                x_Greedy[node_inc_max][item_inc_max] = 1
                phi_Greedy[node_inc_max][item_inc_max] = np.zeros(self.number_of_nodes)
                #print('Delta_evi = '+str(Delta_evi))
                print('Iter '+str(iter)+' Cache pair '+str((node_inc_max,item_inc_max))+', decrease: '+str(Delta_evi[(node_inc_max,item_inc_max)])) 
                #    + ', aggregated cache hit:'+str(self.statistics['CacheHit'][node_inc_max][item_inc_max])\
                #        + ', cache miss:'+str(self.statistics['CacheMiss'][node_inc_max][item_inc_max]))

            # after greedy, update variables (only the caching, since routing is assumed to sum to 1 in Greedy)
            for node in self.Nodes:
                for item in self.Items:
                    self.Nodes[node]['CacheVar'][item][0] = x_Greedy[node][item]
                    self.Nodes[node]['RouteVar'][item] = phi_Greedy[node][item].copy()
                    #self.Nodes[node]['RouteVar'][item] = np.zeros(self.number_of_nodes)
                    #if x_Greedy[node][item] == 0 and NextNode[node][item]:
                    #    self.Nodes[node]['RouteVar'][item][NextNode[node][item][0]] = 1.0
            print('Greedy Done.')

    def update_process_CostGreedy(self):
        """ A heuristic cachesize and content allocation method.
        for each slot, find the (node,item) pair with max aggregated miss cost, then add the node's cache size by 1, to cache the item. """
        Stop_Consec = inf             # will stop if total cost increase consecutive Stop_Consec periods
        TotalCost_prev = np.inf
        FlowCount_prev = {}
        CacheMissCost_prev = {}
        for e in self.EdgesEnumerator:
            FlowCount_prev[e] = 0.0
        for node in self.Nodes:
            CacheMissCost_prev[node] = [0.0] * self.number_of_items

        Inc_Consec = 0
        while True:
            # first, for each slot, add cache size by 1 at the node with largest cache miss cost in last period
            for slot_id in range(1):
                yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)
                MaxCacheMissCost = 0.0
                MaxCostNode = None
                for node in self.Nodes:
                    for item in self.Items:
                        Cost_diff = self.statistics['CacheMissCost'][node][item] - CacheMissCost_prev[node][item]
                        CacheMissCost_prev[node][item] = self.statistics['CacheMissCost'][node][item]
                        if Cost_diff >= MaxCacheMissCost:
                            MaxCacheMissCost = Cost_diff
                            MaxCostNode = (node,item)
                print('MaxCacheMissCostNode = '+str(MaxCostNode)+', cache miss cost= '+str(MaxCacheMissCost))
                if MaxCostNode is not None:
                    self.Nodes[MaxCostNode[0]]['Cache'].size[0] += 1
                    self.Nodes[MaxCostNode[0]]['CacheVar'][MaxCostNode[1]][0] = 1.0
                    self.Nodes[MaxCostNode[0]]['RouteVar'][MaxCostNode[1]] = np.zeros(self.number_of_nodes)
                    
                for node in self.Nodes:
                    self.optCache_mandate_shuffle(node)
            # then, compare the total cost of current preiod with previous period, stop if decrease
            TotalCost_curr = 0.0
            for e in self.EdgesEnumerator:
                FlowCount_new_e = self.Edges[e[0]][e[1]]['FlowCount']
                Flow_e = (FlowCount_new_e - FlowCount_prev[e]) / (self.SimConfig.T_slot * self.SimConfig.L_update)
                FlowCount_prev[e] = FlowCount_new_e
                TotalCost_curr += self.LinkCostCal(e,Flow_e)
            for node in self.Nodes:
                for cachetype in self.CacheTypes:
                    cachesize = self.Nodes[node]['Cache'].GetSize(cachetype)
                    TotalCost_curr += self.CacheCostCal(node,cachetype,cachesize)
            if TotalCost_curr > TotalCost_prev:
                Inc_Consec += 1
            else:
                Inc_Consec = 0
            if Inc_Consec >= Stop_Consec:
                break
            TotalCost_prev = TotalCost_curr

    def update_process_ACR(self):
        """ Adaptive caching with source routing (cf. Stratis).
        Solve using relaxed equivalent LP. """
        pass

    def cacheSize_process_Uniform(self):
        """ Increase cache size by 1 at all nodes each period."""
        Stop_Consec = inf             # will stop if total cost increase consecutive Stop_Consec periods
        TotalCost_prev = np.inf
        FlowCount_prev = {}
        for e in self.EdgesEnumerator:
            FlowCount_prev[e] = 0.0

        if self.AlgoConfig.CacheTypeAlgo != 'All-first':
            print('ERROR: Uniform cache size must be with all-first cache type')
            exit()

        Inc_Consec = 0
        while True:
            yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)
            
            for node in self.Nodes:
                self.Nodes[node]['Cache'].size[0] += 1

            print('Uniformly added cache size by 1')

            TotalCost_curr = 0.0
            for e in self.EdgesEnumerator:
                FlowCount_new_e = self.Edges[e[0]][e[1]]['FlowCount']
                Flow_e = (FlowCount_new_e - FlowCount_prev[e]) / (self.SimConfig.T_slot * self.SimConfig.L_update)
                FlowCount_prev[e] = FlowCount_new_e
                TotalCost_curr += self.LinkCostCal(e,Flow_e)
            for node in self.Nodes:
                for cachetype in self.CacheTypes:
                    cachesize = self.Nodes[node]['Cache'].GetSize(cachetype)
                    TotalCost_curr += self.CacheCostCal(node,cachetype,cachesize)
            if TotalCost_curr > TotalCost_prev:
                Inc_Consec += 1
            else:
                Inc_Consec = 0
            if Inc_Consec >= Stop_Consec:
                break
            TotalCost_prev = TotalCost_curr              

    def cacheSize_process_MaxHit(self):
        """A heuristic cachesize allocation method.
            At each slot, add cache size by 1 at the node with most cache miss.
            Stop at the end of a period, if total cost of this period is larger than previous one."""
        Stop_Consec = inf             # will stop if total cost increase consecutive Stop_Consec periods
        TotalCost_prev = np.inf
        FlowCount_prev = {}
        CacheMiss_prev = {}
        for e in self.EdgesEnumerator:
            FlowCount_prev[e] = 0.0
        for node in self.Nodes:
            CacheMiss_prev[node] = [0.0] * self.number_of_items

        Inc_Consec = 0
        while True:
            # first, for each slot, add cache size by 1 at the node with most cache miss in last slot
            for slot_id in range(self.SimConfig.L_update):
                yield self.env.timeout(self.SimConfig.T_slot)
                MaxCacheMiss = 0.0
                MaxCacheMissNode = None
                for node in self.Nodes:
                    for item in self.Items:
                        CacheMiss_diff = self.statistics['CacheMiss'][node][item] - CacheMiss_prev[node][item]
                        CacheMiss_prev[node][item] = self.statistics['CacheMiss'][node][item]
                        if CacheMiss_diff >= MaxCacheMiss:
                            MaxCacheMiss = CacheMiss_diff
                            MaxCacheMissNode = (node,item)
                print('MaxCacheMissPos = '+str(MaxCacheMissNode)+', cache miss = '+str(MaxCacheMiss))
                if MaxCacheMissNode is not None:
                    self.Nodes[MaxCacheMissNode[0]]['Cache'].size[0] += 1
                
            # then, compare the total cost of current preiod with previous period, stop if decrease
            TotalCost_curr = 0.0
            for e in self.EdgesEnumerator:
                FlowCount_new_e = self.Edges[e[0]][e[1]]['FlowCount']
                Flow_e = (FlowCount_new_e - FlowCount_prev[e]) / (self.SimConfig.T_slot * self.SimConfig.L_update)
                FlowCount_prev[e] = FlowCount_new_e
                TotalCost_curr += self.LinkCostCal(e,Flow_e)
            for node in self.Nodes:
                for cachetype in self.CacheTypes:
                    cachesize = self.Nodes[node]['Cache'].GetSize(cachetype)
                    TotalCost_curr += self.CacheCostCal(node,cachetype,cachesize)
            if TotalCost_curr > TotalCost_prev:
                Inc_Consec += 1
            else:
                Inc_Consec = 0
            if Inc_Consec >= Stop_Consec:
                break
            TotalCost_prev = TotalCost_curr

    def cacheSize_process_MinCost(self):
        """Another heuristic cache allocation method, an augmentation of MaxHit.
            At each period, add cache size by 1 at the node with largest cache miss cost.
            Cache miss cost is calculated as (1) average travel hop number of all responses traveled by, or
            (2) average sum of marginal link cost at F=0 of all responses traveled by.
            Stop at the end of a period, if total cost of this period is larger than previous one."""
        Stop_Consec = inf             # will stop if total cost increase consecutive Stop_Consec periods
        TotalCost_prev = np.inf
        FlowCount_prev = {}
        CacheMissCost_prev = {}
        for e in self.EdgesEnumerator:
            FlowCount_prev[e] = 0.0
        for node in self.Nodes:
            #CacheMissCost_prev[node] = [0.0] * self.number_of_items
            CacheMissCost_prev[node] = 0.0

        Inc_Consec = 0
        while True:
            # first, for each slot, add cache size by 1 at the node with largest cache miss cost in last slot
            yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)
            #for slot_id in range(self.SimConfig.L_update):
            #    yield self.env.timeout(self.SimConfig.T_slot)
            #    MaxCacheMissCost = 0.0
            #    MaxCostNode = None
            #    for node in self.Nodes:
            #        for item in self.Items:
            #            Cost_diff = self.statistics['CacheMissCost'][node][item] - CacheMissCost_prev[node][item]
            #            CacheMissCost_prev[node][item] = self.statistics['CacheMissCost'][node][item]
            #            if Cost_diff >= MaxCacheMissCost:
            #                MaxCacheMissCost = Cost_diff
            #                MaxCostNode = (node,item)
            #    print('MaxCacheMissCostNode = '+str(MaxCostNode)+', miss cost= '+str(MaxCacheMissCost))
            #    if MaxCostNode is not None:
            #        self.Nodes[MaxCostNode[0]]['Cache'].size[0] += 1

            MaxCacheMissCost = 0.0
            MaxCostNode = None
            for node in self.Nodes:
                # add according to (node,item) pair
                #for item in self.Items:
                #    Cost_diff = self.statistics['CacheMissCost'][node][item] - CacheMissCost_prev[node][item]
                #    CacheMissCost_prev[node][item] = self.statistics['CacheMissCost'][node][item]
                #    if Cost_diff >= MaxCacheMissCost:
                #        MaxCacheMissCost = Cost_diff
                #        MaxCostNode = (node,item)

                # add only according to node
                Cost_diff = sum(self.statistics['CacheMissCost'][node]) - CacheMissCost_prev[node]
                CacheMissCost_prev[node] = sum(self.statistics['CacheMissCost'][node])
                if Cost_diff >= MaxCacheMissCost:
                    MaxCacheMissCost = Cost_diff
                    MaxCostNode = node
            print('MaxCacheMissCostNode = '+str(MaxCostNode))#+', miss cost= '+str(MaxCacheMissCost))
            if MaxCostNode is not None:
                self.Nodes[MaxCostNode]['Cache'].size[0] += 1
                
            # then, compare the total cost of current preiod with previous period, stop if decrease
            TotalCost_curr = 0.0
            for e in self.EdgesEnumerator:
                FlowCount_new_e = self.Edges[e[0]][e[1]]['FlowCount']
                Flow_e = (FlowCount_new_e - FlowCount_prev[e]) / (self.SimConfig.T_slot * self.SimConfig.L_update)
                FlowCount_prev[e] = FlowCount_new_e
                TotalCost_curr += self.LinkCostCal(e,Flow_e)
            for node in self.Nodes:
                for cachetype in self.CacheTypes:
                    cachesize = self.Nodes[node]['Cache'].GetSize(cachetype)
                    TotalCost_curr += self.CacheCostCal(node,cachetype,cachesize)
            if TotalCost_curr > TotalCost_prev:
                Inc_Consec += 1
            else:
                Inc_Consec = 0
            if Inc_Consec >= Stop_Consec:
                break
            TotalCost_prev = TotalCost_curr

    def LinkCostCal(self,e,F):   
        """ Calculate link cost Dij(Fij) on link e with given flow F."""
        if F < eps:
            return 0.0
        elif e not in self.EdgesEnumerator:
            return inf
        
        link_para = self.G[e[0]][e[1]]['LinkPara']

        if self.LinkCostType == 'Linear':                           # for linear cots, D = w * F
            return F * link_para

        elif self.LinkCostType == 'Queue':                          # for queueing delay, D = F/(C-F)
            if F < link_para:
                return F / (link_para - F)
            else:
                return inf

        elif self.LinkCostType == 'Quadratic':                      # quadratic cost: D = w * F^2 + w*F
            return link_para * (F + F **2)
        
        elif self.LinkCostType == 'Taylor':                      # 3-order Taylor expansion: D = w*F + w^2*F^2 + w^3*F^3
            return (link_para * F + link_para**2 * F**2 + link_para**3 * F**3)
        else:
            print('ERROR: Undefined link cost type.')
            exit()
    def LinkMargCal(self,e,F):
        """ Calculate the marginal link cost D'ij(Fij) on link e with given flow F."""
        if e not in self.EdgesEnumerator:
            return inf
        
        link_para = self.G[e[0]][e[1]]['LinkPara']
        if self.LinkCostType == 'Linear':                           # for linear cost, D' = w
            return link_para
        elif self.LinkCostType == 'Queue':                          # for queueing delay, D' = C/(C-F)^2
            if F < link_para:
                return link_para / (link_para - F)**2
            else:
                return inf
        elif self.LinkCostType == 'Quadratic':                      # quadratic cost: D' = 2w * F + w
            return 2.0* link_para * F + link_para
        elif self.LinkCostType == 'Taylor':                      # 3-order Taylor expansion: D = w*F + w^2*F^2 + w^3*F^3
            return (link_para + 2* link_para**2 * F + 3* link_para**3 * F**2)
        else:
            print('ERROR: Undefined link cost type.')
            exit()

    def CompCostCal(self,node,workload):
        """computation cost Ci(Gi) calculation given the workload Gi"""
        comp_para = self.G.nodes[node]['CompPara']
        if self.CompCostType == 'Linear':
            return comp_para * workload
        elif self.CompCostType == 'Queue':
            if workload < comp_para:
                return workload / (comp_para - workload)
            else:
                return inf
        elif self.CompCostType == 'Taylor':
            return (comp_para * workload + comp_para**2 * workload**2 + comp_para**3 * workload**3)
        else:
            print('ERROR: Undefined comp cost type.')
            exit()

    def CompMargCal(self,node,workload):
        """Marginal computation cost Ci'(Gi)"""
        comp_para = self.G.nodes[node]['CompPara']
        if self.CompCostType == 'Linear':
            return comp_para
        elif self.CompCostType == 'Queue':
            if workload < comp_para:
                return comp_para / (comp_para - workload)
            else:
                return inf
        elif self.CompCostType == 'Taylor':
            return (comp_para + 2 * comp_para**2 * workload + 3 * comp_para**3 * workload**2)
        else:
            print('ERROR: Undefined comp cost type.')
            exit()

    def CacheCostCal(self,node,CacheSize):
        """ Calculate B_i(Y_i), the cache cost at a node. 
        Cache size is specified by input. """
        cache_para = self.G.nodes[node]['CachePara']

        if self.CacheCostType == 'Linear':
            return cache_para * CacheSize
        elif self.CacheCostType == 'Quadratic':
            return cache_para * (CacheSize**2 + CacheSize)
        elif self.CacheCostType == 'Capacity':             # using approximation D(Y) = 1/[ Scl * (CacheCap - Y)], and Scl should be large, here cachecap = CachePara_m
            #Scl = 1e2
            tolerence = 2e-2                            # if cache var is not exceeding this plus capacity, it's fine
            #return 1.0 / (Scl * (CachePara_m - CacheSize_m)) if CacheSize_m < CachePara_m else np.inf
            return 0.0 if CacheSize < cache_para + tolerence else np.inf
        else:
            print('ERROR: Undefined cache cost type.')
            exit()

    def CacheMargCal(self,node,CacheSize):
        """ Calculate B'_i(Y_i), the marginal cache cost. 
        Cache size is specified by input. """
        #CacheSize_m = self.Nodes[node]['Cache'].GetSize(m)
        cache_para = self.G.nodes[node]['CachePara']

        if self.CacheCostType == 'Linear':
            return cache_para 
        elif self.CacheCostType == 'Quadratic':
            return 2.0 * cache_para * CacheSize + CacheSize
        elif self.CacheCostType == 'Capacity':             # derivative of approximation, D'(Y) = 1/ [Scl * (CacheCap - Y)^2]
            Scl = 20
            return 1.0 / (Scl * (cache_para - CacheSize)**2) if CacheSize < cache_para else np.inf
        else:
            print('ERROR: Undefined cache cost type.')
            exit()

    def LinkCapacityCal(self):
        """ Generate the network's link capacity, according to the link cost type. Used to generate initial routing var, and check feasibility.
            For linear cost, link capacity is infinity.
            For queueing delay, link capacity is 0.9 * LinkPara"""
        print('Generating link capacity...')
        LinkSaturateFactor = 0.8
        LinkCapacity = {}
        if self.LinkCostType == 'Linear' or self.LinkCostType == 'Quadratic' or self.LinkCostType == 'Taylor':
            sumrates = 0.0
            for i in self.Nodes:
                for k in self.Items:
                    for s in self.Stages:
                        sumrates += self.RequestRates[i][k][s] * self.PacketSize[k][s]
        #print(self.EdgesEnumerator)
        for i in self.Nodes:
            LinkCapacity[i] = {}
            for j in self.Nodes:
                #print((i,j))
                if (i,j) in self.EdgesEnumerator:
                    if self.LinkCostType == 'Linear' or self.LinkCostType == 'Quadratic' or self.LinkCostType == 'Taylor':
                        LinkCapacity[i][j] = sumrates
                    elif self.LinkCostType == 'Queue':
                        #print(self.G[i][j]['LinkPara'])
                        LinkCapacity[i][j] = LinkSaturateFactor * self.G[i][j]['LinkPara']
                    else:
                        print('ERROR: undefined link cost type')
                else:
                    LinkCapacity[i][j] = 0
        return LinkCapacity

    def CompCapacityCal(self):
        """Generate computation workload upperbounds for each node. """
        print('Generating computation workload limits...')
        CompSaturateFactor = 0.8
        CompCapacity = {}   # dict node:capacity
        if self.CompCostType == 'Linear' or self.CompCostType == 'Quadratic' or self.CompCostType == 'Taylor':
            for node in self.Nodes:
                CompCapacity[node] = inf
        elif self.CompCostType == 'Queue':
            for node in self.Nodes:
                CompCapacity[node] = CompSaturateFactor * self.G[node]['CompPara']
        return CompCapacity

    def RouteVarInitCal(self):
        """ Calculate an initial routing variable that satisfies the link and computation capacity.
            Detail calculation is specified by config file.
            Note: the initial route variable shoule be loop-free"""
        RouteVarInit = {}         # RouteVarInit[i][k][s] for node i item k and stage s is a list storing phi_ij(s) for all j = 0,1,..,N. (0 represents local computation)
        for i in self.Nodes:
            RouteVarInit[i] = {}    
            for k in self.Items:
                RouteVarInit[i][k] = {}
                for s in self.Stages:
                    RouteVarInit[i][k][s] = np.zeros(self.number_of_nodes + 1)

        print('Generating initial routing scheme...')

        if self.AlgoConfig.RouteInitType == 1:  
            # shortest path using LP at F=0, with no caching
            # min w.*f, s.t. Af <= b, A_eq f = b_eq
            # f is the flow f_ij(s), s is all stages, j is from 0 to N
            # w is the link marginal at F=0 and the comp cost marginal at C=0, both weighted by packet size and comp workload
            # A and b are for link capacity and non-negativity, A_eq and b_eq are for flow conservation
            length_f = ( self.number_of_nodes + self.number_of_edges) * self.number_of_items * self.number_of_stages  # dimension of f_ij(k,s). Ordered by j,k,s. j starts from 0.
            w = np.zeros(length_f)  # set vector w
            # first assign w entries for computation workload
            for node in self.Nodes:
                comp_marg = self.CompMargCal(node,workload=0.0)
                for item in self.Items:
                    for stage in self.Stages:
                        pos_comp = node * self.number_of_stages * self.number_of_items + item * self.number_of_stages + self.stage_to_num(stage)     # node starts from 0
                        w[pos_comp] = self.CompWorkload[stage] * comp_marg
            # then assign w for link costs
            for e_id in range(self.number_of_edges):
                e = self.EdgesEnumerator[e_id]
                link_marg = self.LinkMargCal(e,F=0.0)
                for item in self.Items:
                    for stage in self.Stages:
                        pos_flow = (self.number_of_nodes + e_id) * self.number_of_stages * self.number_of_items + item * self.number_of_stages + self.stage_to_num(stage) 
                        w[pos_flow] = self.PacketSize[item][stage] * link_marg

            n_cons_neq = self.number_of_nodes + self.number_of_edges   # number of neq constraints. The comp/link capacities.
            # \sum_{k} f_ij(k) <= Cij, for all i,j
            A = np.zeros((n_cons_neq,length_f))
            b = np.zeros(n_cons_neq)
            # first computation capacities
            for node in self.Nodes:
                pos_const = node
                comp_cap = self.CompCapacity[node]
                if not isinf(comp_cap):
                    b[pos_const] = comp_cap
                    #print("b[pos_const] = "+ str(isinf(b[pos_const])))
                else:
                    b[pos_const] = 1e6
                for item in self.Items:
                    for stage in self.Stages:
                        pos_comp = node * self.number_of_stages * self.number_of_items + item * self.number_of_stages + self.stage_to_num(stage) 
                        A[pos_const,pos_comp] = self.CompWorkload[stage]
            # then link capacities
            for e_id in range(self.number_of_edges):
                e = self.EdgesEnumerator[e_id]
                link_cap = self.LinkCapacity[e[1]][e[0]]
                pos_const = self.number_of_nodes + e_id
                b[pos_const] = link_cap
                for item in self.Items:
                    for stage in self.Stages:
                        pos_flow = (self.number_of_nodes + e_id) * self.number_of_stages * self.number_of_items + item * self.number_of_stages + self.stage_to_num(stage) 
                        A[pos_const,pos_flow] = self.PacketSize[item][stage]

            n_cons_eq = self.number_of_nodes * self.number_of_items * self.number_of_stages # number of eq constraints. Flow conservation for each node, each item, each stage
            Aeq = np.zeros((n_cons_eq,length_f))
            beq = np.zeros(n_cons_eq)
            for node in self.Nodes:
                for item in self.Items:
                    for stage in self.Stages:
                        pos_const = node * self.number_of_items * self.number_of_stages + item * self.number_of_stages + self.stage_to_num(stage) 
                        # sum_j fijks = 0, if stage = 0 and i is server of k
                        # sum_j fijks - sum_j fjiks = ris, if stage is len(app)
                        # sum_j fijs - sum_j fjis = 0, o.w.

                        # out going flows
                        pos_comp = node * self.number_of_items * self.number_of_stages + item * self.number_of_stages + self.stage_to_num(stage) 
                        Aeq [pos_const,pos_comp] = 1.0
                        for j in self.Nodes:
                            if (node,j) in self.EdgesEnumerator:
                                e_id = self.EdgesEnumerator.index((node,j))
                                pos_flow = (self.number_of_nodes + e_id) * self.number_of_items * self.number_of_stages + item * self.number_of_stages + self.stage_to_num(stage) 
                                Aeq [pos_const,pos_flow] = 1.0
                        # in going flows
                        if stage[1] == 0 and node == self.DesServers[item]:  # if is for data at server, no in-going constraints
                            pass
                        elif stage[1] == 0:   # if is data, no in-going computation flows, just links
                            for j in self.Nodes:
                                if (j,node) in self.EdgesEnumerator:
                                    e_id = self.EdgesEnumerator.index((j,node))
                                    pos_flow = (self.number_of_nodes + e_id) * self.number_of_items * self.number_of_stages + item * self.number_of_stages + self.stage_to_num(stage) 
                                    Aeq [pos_const,pos_flow] = -1.0
                        else:   # other stages, accounts for the in-going computation results
                            stage_prev = (stage[0],stage[1]-1)
                            pos_comp_prev = node * self.number_of_items * self.number_of_stages + item * self.number_of_stages + self.stage_to_num(stage_prev) 
                            Aeq [pos_const,pos_comp_prev] = -1.0
                            for j in self.Nodes:
                                if (j,node) in self.EdgesEnumerator:
                                    e_id = self.EdgesEnumerator.index((j,node))
                                    pos_flow = (self.number_of_nodes + e_id) * self.number_of_items * self.number_of_stages + item * self.number_of_stages + self.stage_to_num(stage) 
                                    Aeq [pos_const,pos_flow] = -1.0
                        #if stage[1] ==  self.AppChain_Len[stage[0]]: # if is final stage (request inputs), account for the input rates
                        beq[pos_const] = self.RequestRates[node][item][stage]
                        #if stage[1] == self.AppChain_Len[stage[0]] and node == self.DesServers
                        #if stage[1] >= 1:   # if not raw data, count the ingoing traffic

            # linear programming solver
            #print('A = '+str(A))
            #print('b = '+str(b))
            #print('Aeq = '+str(Aeq))
            #print('beq = '+str(beq))
            res = sciopt.linprog(w,A,b,Aeq,beq,method = 'revised simplex')
            #LB = np.zeros(length_f)
            #UB = np.full(length_f, np.inf)
            #res = optimize.linprog(w,A,b,Aeq,beq,LB,UB)

            #print(self.LinkCapacity)
            if not res.success:
                print('ERROR: Optimal solution not found.')
                exit()
            f_opt = res.x

            # calculate t_i(k,s), ordered by i,k,s
            length_t = self.number_of_nodes * self.number_of_items * self.number_of_stages
            t_opt = np.zeros(length_t)
            for node in self.Nodes:
                for item in self.Items:
                    for stage in self.Stages:
                        t_pos = node * self.number_of_items * self.number_of_stages + item * self.number_of_stages + self.stage_to_num(stage) 
                        # t_i(k,s) consists of all ingoing flows (comp and flow), as well as input rate
                        if stage[1] >= 1:   
                            stage_prev = (stage[0],stage[1]-1)
                            pos_comp_prev = node * self.number_of_stages * self.number_of_items + item * self.number_of_stages + self.stage_to_num(stage_prev) 
                            t_opt[t_pos] += f_opt[pos_comp_prev]
                            for j in self.Nodes:
                                if (j,node) in self.EdgesEnumerator:
                                    e_id = self.EdgesEnumerator.index((j,node))
                                    pos_flow = (self.number_of_nodes + e_id) * self.number_of_items * self.number_of_stages + item * self.number_of_stages + self.stage_to_num(stage) 
                                    t_opt[t_pos] += f_opt[pos_flow]
                            t_opt[t_pos] += self.RequestRates[node][item][stage]
                
            # calculate corrsponding phi. For i,k,s with t_i(k,s) = 0 and s is not stage 0, put all into computation; if s is 0, find shortest path to server
            for node in self.Nodes:
                for item in self.Items:
                    for stage in self.Stages:
                        if stage[1] == 0 and node in self.DesServers[item]: # if stage 0 and server, set all phi to 0
                            pass
                        else:   # otherwise, the sum of phi should be 1
                            t_pos = node * self.number_of_items * self.number_of_stages + item * self.number_of_stages + self.stage_to_num(stage) 
                            t_iks = t_opt[t_pos]
                            if  t_iks > eps:   # for i,k,s with t_i(k,s) > 0, phi_ij(k,s) = f_ij(k,s)/t_i(k,s)
                                pos_comp = node * self.number_of_stages * self.number_of_items + item * self.number_of_stages + self.stage_to_num(stage) 
                                g_iks = f_opt[pos_comp]
                                RouteVarInit[node][item][stage][0] = g_iks / t_iks  # computation
                                for j in self.Nodes[i]['Neighbor']:
                                    e = (i,j)
                                    e_id = self.EdgesEnumerator.index(e)
                                    pos_flow = (self.number_of_nodes + e_id) * self.number_of_items * self.number_of_stages + item * self.number_of_stages + self.stage_to_num(stage) 
                                    f_ijks = f_opt[pos_flow]
                                    RouteVarInit[i][k][j + 1] = max(f_ijks / t_iks,0) # note: since 0 is for computation
                            else:               # for i,k,s with t_i(k,s) = 0
                                if stage[1] > 0:   # if not stage 0, assign all to comp
                                    RouteVarInit[node][item][stage][0] = 1.0
                                else:
                                    sp = nx.shortest_path(self.G,node,self.DesServers[item][0])
                                    if len(sp) >= 2:    # if the shortest path is of length more that two, route to the first hop. Otherwise, i is a source of k, all phi are 0
                                        RouteVarInit[node][item][stage][sp[1] +1] = 1.0
            
        return RouteVarInit

    def TheoreticalFlowCal(self):
        """ Calculate the theoretical flow from input rate r and current routing variables. 
            By solving a set of linear equations about t, and use f = t * phi.
            Output is a tuple of 3 dicts: ( { e : flow_theo_e}, { i : workload_theo_i }, {k: s: [list of t_i(k,s) for all i]} ). """

        Flow_theo = {}
        Wokload_theo = {}
        for e in self.EdgesEnumerator:
            Flow_theo[e] = 0.0
        for i in self.Nodes:
            Wokload_theo[i] = 0.0
        traffic = {}
        for k in self.Items:
            traffic[k] = {}
            for s in self.Stages:
                traffic[k][s] = []
            
        # construct the linear equations: t_iks - sum_{j = 1..N} t_jks * \phi ji(ks) = riks + tik(s+1) * \phi0k(s+1)
        n_eq = self.number_of_nodes # number of equations for each (k,s)
        for k in self.Items:
            for app in range(self.number_of_applications):   # the stage has to start from final stages, so no computation input is involved
                for task in range(self.AppChain_Len[app],-1,-1):
                    stage = (app,task)
                    length_t = self.number_of_nodes
                    A = np.zeros((n_eq,length_t))
                    b = np.zeros(n_eq)
                    for i in self.Nodes:
                    #    print('i = '+str(i)+', neighbor = '+str(self.Nodes[i]['Neighbor']))
                        A[i][i] = 1.0
                        for j in self.Nodes[i]['Neighbor']:
                            A[i][j] = -1.0 * self.Nodes[j]['RouteVar'][k][stage][i+1] if self.Nodes[j]['RouteVar'][k][stage][i+1] > eps else 0.0
                        if task == self.AppChain_Len[app]:    # if final stage, no computational input
                            b[i] = self.RequestRates[i][k][stage]
                        else:       # accont for computational input
                            stage_next = (stage[0],stage[1]+1)
                            b[i] = self.RequestRates[i][k][stage] + traffic[k][stage_next][i] * self.Nodes[i]['RouteVar'][k][stage_next][0]
                    #print('A = '+str(A))
                    #print('b = '+str(b))
                    f_ks = np.linalg.solve(A, b)
                    #print('f_ks = '+str(f_ks))
                    traffic[k][stage] = f_ks[:]
                    #print('f_k = '+str(f_k))
                    for e in self.EdgesEnumerator:
                        #print('e = '+str(e))
                        #print('k = '+str(k))
                        #print('stage = '+str(stage))
                        Flow_theo[e] += f_ks[e[1]] * self.Nodes[e[1]]['RouteVar'][k][stage][e[0]+1] * self.PacketSize[k][stage]    # note: ti * phiij is accounted for the reversed link
                    for i in self.Nodes:
                        #print('self.Nodes[i][RouteVar][k][stage] = ' +str(self.Nodes[i]['RouteVar'][k][stage]))
                        Wokload_theo[i] += f_ks[i] * self.Nodes[i]['RouteVar'][k][stage][0] * self.CompWorkload[stage]   # computation workload

        return (Flow_theo, Wokload_theo, traffic)

    def TheoreticalFlowCal_GivenPhi(self,Phi):
        """ Calculate the theoretical flow from input rate r using a given routing variable Phi. 
            By solving a set of linear equations about t, and use f = t * phi.
            Note: assume the sum of phi = 1
            input: Phi is a dict of dict {node i: item k : [array of phi_ij(k) for all j]}
            Output is a dict of dict {edge e: item k: f_ek} """

        flow_theo = {}
        for e in self.EdgesEnumerator:
            flow_theo[e] = {}
            for item in self.Items:
                flow_theo[e][item] = 0.0
 
         # construct the linear equations
        n_eq = self.number_of_nodes
        for k in self.Items:
            # compute [t_i] for each item, solving linear set: A*t = b: t_i - \sum_{j in N_i} \phi_ji * t_j = r_i for all i
            length_t = self.number_of_nodes
            A = np.zeros((n_eq,length_t))
            b = np.zeros(n_eq)
            for i in self.Nodes:
            #    print('i = '+str(i)+', neighbor = '+str(self.Nodes[i]['Neighbor']))
                A[i][i] = 1.0
                for j in self.Nodes[i]['Neighbor']:
                    #A[i][j] = -1.0 * self.Nodes[j]['RouteVar'][k][i] if self.Nodes[j]['RouteVar'][k][i] > eps else 0.0
                    A[i][j] = -1.0 * Phi[j][k][i] if Phi[j][k][i] > eps else 0.0
                b[i] = self.RequestRates[i][k]
            #print('A = '+str(A))
            #print('b = '+str(b))
            f_k = np.linalg.solve(A, b)
            #print('f_k = '+str(f_k))
            for e in self.EdgesEnumerator:
            #    print('e = '+str(e[0]) + ',' +str(e[1]))
                flow_theo[e][k] = f_k[e[1]] * Phi[e[1]][k][e[0]]
        return flow_theo

    def TheoreticalCostCal(self):
        """Calculate the current theoretical link costs, computation costs and cache costs, from the input rates, routing and caching variables.
            Returns a tuple (Linkcost,CompCost,Cachecost)."""
        LinkCost_Theo = 0.0
        CompCost_Theo = 0.0
        CacheCost_Theo = 0.0
        (Flow_theo,Workload_theo,traffic) = self.TheoreticalFlowCal()
        #print('TheoreticalCostCal : sum_Flow_theo = '+str(sum(Flow_theo.values())))
        for e in self.EdgesEnumerator:
            Flow_theo_e = Flow_theo[e]
            Cost_theo_e = self.LinkCostCal(e,Flow_theo_e)
            LinkCost_Theo += Cost_theo_e
        for i in self.Nodes:
            Workload_theo_i = Workload_theo[i]
            CompCost_theo_i = self.CompCostCal(i,Workload_theo_i)
            CompCost_Theo += CompCost_theo_i
        for i in self.Nodes:
            Cachesize_theo_i = 0.0
            for k in self.Items:
                for s in self.Stages:
                    Cachesize_theo_i += self.Nodes[i]['CacheVar'][k][s]
            CacheCost_theo_i = self.CacheCostCal(i,Cachesize_theo_i)
            CacheCost_Theo += CacheCost_theo_i
        return (LinkCost_Theo,CompCost_Theo,CacheCost_Theo)

    def TheoreticalLinkCostCal_GivenPhi(self,Phi):
        """ Calculate theoretical sum link cost for a given phi."""
        LinkCost_Theo = 0.0
        flow_theo = self.TheoreticalFlowCal_GivenPhi(Phi)
        for e in self.EdgesEnumerator:
            Flow_theo_e = sum(flow_theo[e].values())
            Cost_theo_e = self.LinkCostCal(e,Flow_theo_e)
            LinkCost_Theo += Cost_theo_e
        return LinkCost_Theo

    def runSim(self):
        """" Run simulation for the simulation time in config file."""
        print('Simulation running...')
        total_time = self.SimConfig.T_sim + self.SimConfig.T_extension
        self.env.run(until=total_time)
        print('Simulation runtime up.')
    
    def print_time_process(self,T_TimePrint = 100.0):
        """ Process that prints time while running."""
        while True:
            if self.env.now <= self.SimConfig.T_sim:
                if len(self.statistics['TotalCost_actual']) >= 1:
                    print('Simlutaion time: '+str(self.env.now)+', most recent actual total cost = '+str(self.statistics['TotalCost_actual'][-1])+' , average request delay = '+str(self.statistics['RequestDelay']['Average']))
                else:
                    print('Simlutaion time: '+str(self.env.now))
                yield self.env.timeout(T_TimePrint)
            else:
                print('Simlutaion time: Extension')
                yield self.env.timeout(inf)

    def flow_calculate_process(self,e):
        """ Process that calculates Fij for each link, based on FlowCount. Calculate at the end of every update slot.
            Note that this is used for algorithm (when passing ctrl msg), not result monitor. """
        if self.AlgoConfig.RouteAlgo == 'GP':
            # if using GP, the flow is calculated at the begining of each update slot
            T_start = self.SimConfig.T_slot * (self.SimConfig.L_update -1)
            T_interval = self.SimConfig.T_slot * self.SimConfig.L_update -1
        else:
            # if not GP, update every slot
            T_start = self.SimConfig.T_slot
            T_interval = self.SimConfig.T_slot
        FlowCount_prev = 0.0

        yield self.env.timeout(T_start )
        while True:
            FlowCount_new = self.Edges[e[0]][e[1]]['FlowCount']
            Flow = (FlowCount_new - FlowCount_prev) / T_interval
            self.Edges[e[0]][e[1]]['CurrentFlow'] = Flow
            #if Flow >= self.LinkCapacity[e[0]][e[1]]:
                #print('Warning: Flow '+str(Flow)+' exceeds capacity '+str(self.LinkCapacity[e[0]][e[1]]))
            #if self.AlgoConfig.RouteAlgo == 'GP' and Flow >= self.G[e[0]][e[1]]['LinkPara'] and self.LinkCostType != 'Linear':
            #if Flow >= self.G[e[0]][e[1]]['LinkPara'] and self.LinkCostType != 'Linear':
            #    print(LineBreak)
            #    print('Warning: Edge '+str(e)+', flow '+str(Flow)+' exceeds capacity '+str(self.G[e[0]][e[1]]['LinkPara']))
            #    (Theo_flow,traffic) = self.TheoreticalFlowCal()
            #    print('Theoretical flow is '+str(Theo_flow[e]))
            FlowCount_prev = FlowCount_new
            yield self.env.timeout(T_interval)

    def monitor_process(self):
        """ Process that periodically monitors the network cost.
            Note that the flow on links are evaluated both in actual (time-averaged packet count) and theoretical (calculated by r and phi)"""
        yield self.env.timeout(self.SimConfig.MonitorInitTime)
        FlowCount_prev = {}
        WorkloadCount_prev = {}
        for e in self.EdgesEnumerator:
            FlowCount_prev[e] = self.Edges[e[0]][e[1]]['FlowCount']
        for i in self.Nodes:
            WorkloadCount_prev[i] = self.Nodes[i]['WorkloadCount']
            #print('WorkloadCount =' + str(self.Nodes[i]['WorkloadCount']))
        Monitor_StartTime = time.time()

        while True:
            yield self.env.timeout(self.SimConfig.T_monitor)
            self.statistics['MonitorTime'].append(self.env.now)
            Monitor_NowTime = time.time()
            self.statistics['MonitorRealTime'].append(float(Monitor_NowTime - Monitor_StartTime))
            LinkCost_Actual = 0.0
            CacheCost_Actual = 0.0
            CompCost_Actual = 0.0
            #CacheCost_Theo = 0.0

            # test if the variables sum up to 1
            for node in self.Nodes:
                for item in self.Items:
                    for stage in self.Stages:
                        sum_phi = sum(self.Nodes[node]['RouteVar'][item][stage])
                        y = self.Nodes[node]['CacheVar'][item][stage]
                        if self.AlgoConfig.CacheAlgo == 'GP' and \
                            np.abs(sum_phi + y - 1.0) >= eps and \
                                ( stage[1] != 0 or node not in self.DesServers[item]):
                            print('Warning: variables phi and y sum up to '+str(sum_phi + y)+', at node '+str(node)+' item '+str(item))

            #Flow_theo = self.TheoreticalFlowCal()
            Flow_actual = {}    # dict e : flow
            Workload_actual = {}    # dict i : workload
            for e in self.EdgesEnumerator:  # link costs
                FlowCount_new_e = self.Edges[e[0]][e[1]]['FlowCount']
                
                # link cost by the time averaged packet count
                Flow_actual[e] = (FlowCount_new_e - FlowCount_prev[e]) / self.SimConfig.T_monitor
                Cost_actual_e = self.LinkCostCal(e,Flow_actual[e])
                LinkCost_Actual += Cost_actual_e
                FlowCount_prev[e] = FlowCount_new_e

            for i in self.Nodes:  # comp costs
                WorkloadCount_new_e = self.Nodes[i]['WorkloadCount']
                
                Workload_actual[i] = (WorkloadCount_new_e - WorkloadCount_prev[i]) / self.SimConfig.T_monitor
                #print('Workload_actual[i] = '+str(Workload_actual[i]))
                Compcots_actual_i = self.CompCostCal(i,Workload_actual[i])
                CompCost_Actual += Compcots_actual_i
                WorkloadCount_prev[i] = WorkloadCount_new_e

           # cache cost, calculated from snapshot OccupiedSize()
            CaseSize_theo = {}
            for i in self.Nodes:
                #print('CacheVar = '+str(self.Nodes[i]['CacheVar']))
                CaseSize_theo[i] = sum([ self.Nodes[i]['CacheVar'][k][s] for k in self.Items for s in self.Stages ]  )
                Cachesize_actual = self.Nodes[i]['Cache'].GetSize()
                Cost_actual = self.CacheCostCal(i,Cachesize_actual)
                CacheCost_Actual += Cost_actual

            (LinkCost_Theo,CompCost_Theo,CacheCost_Theo) = self.TheoreticalCostCal()
            self.statistics['LinkCost_actual'].append(LinkCost_Actual)
            self.statistics['LinkCost_theo'].append(LinkCost_Theo)
            self.statistics['CompCost_actual'].append(CompCost_Actual)
            self.statistics['CompCost_theo'].append(CompCost_Theo)
            self.statistics['CacheCost_actual'].append(CacheCost_Actual)
            self.statistics['CacheCost_theo'].append(CacheCost_Theo)
            self.statistics['TotalCost_actual'].append(LinkCost_Actual + CompCost_Actual + CacheCost_Actual)
            self.statistics['TotalCost_theo'].append(LinkCost_Theo + CompCost_Theo + CacheCost_Theo)

            if self.SimConfig.Is_draw_network is True:
                # draw a network figure to illustrate the flow and cache size in the network
                maxFlow = 100.0
                maxSize = max(CaseSize_theo.values())
                print('maxSize = '+str(maxSize))
                G_draw = nx.Graph()
                for i in self.Nodes:
                    dim = int(np.sqrt(self.number_of_nodes))
                    G_draw.add_node(i ) # size of node is the cache size
                    G_draw.nodes[i]['pos'] = (i//dim, i- dim*(i//dim))    # positioning, only work for 6x6 grid
                for e in self.EdgesEnumerator:
                    if e not in G_draw.edges():
                        G_draw.add_edge(e[0],e[1], weight = (Flow_actual[e]+Flow_actual[(e[1],e[0])] + 1) / maxFlow)    # width of link is the flow (both direction)
                pos=nx.get_node_attributes(G_draw,'pos')
                fig = plt.figure()
                #nx.draw(G_draw, pos, ax=fig.add_subplot(111) ,width='weight')
                sizescale = 100.0
                widthscale = 20.0
                nodesize = [sizescale*(CaseSize_theo[i] +0.1) for i in self.Nodes]
                weights = [widthscale*(G_draw[u][v]['weight']) for u,v in G_draw.edges()]
                nx.draw(G_draw, pos, node_size = nodesize, ax=fig.add_subplot(111), node_color = rgb2hex(237,125,49), edge_color = rgb2hex(91,155,213),width=weights)
                #plt.savefig('./NetworkMonitor '+str(self.env.now)+'.png')
                fig.savefig('./MonitorDraws/MonitorDraw '+str(self.env.now)+'.png')
                plt.close(fig)
        

    def Is_looped(self):
        """Check if current routing scheme is loop-free, returns 1 if there is loop, 0 if not."""
        Is_Looped = False
        for k in self.Items:
            for s in self.Stages:
                DAG_k = nx.DiGraph()
                DAG_k.add_nodes_from(self.Nodes.keys())
                for i in self.Nodes:
                    for j in self.Nodes[i]['Neighbor']:
                        if self.Nodes[i]['RouteVar'][k][s][j+1] > eps:
                            DAG_k.add_edge(i,j)
                if not nx.is_directed_acyclic_graph(DAG_k):
                    Is_Looped = True
                    break
        return Is_Looped

    def Is_looped_GivenPhi(self,RouteVar):
        """Check if the given routing scheme is loop-free, returns 1 if there is loop, 0 if not.
        Input: RouteVar: { node i: item k: [array of phi_ijk for all j]}"""
        Is_Looped = False
        for k in self.Items:
            DAG_k = nx.DiGraph()
            DAG_k.add_nodes_from(self.Nodes.keys())
            for i in self.Nodes:
                for j in self.Nodes[i]['Neighbor']:
                    if RouteVar[i][k][j] > eps:
                        DAG_k.add_edge(i,j)
            if not nx.is_directed_acyclic_graph(DAG_k):
                Is_Looped = True
                break
        return Is_Looped

    def printStatistics(self, OutFile = sys.stdout):
        """ Print the statistics to output file. Default to screen"""
        print(LineBreak+'Simulation statistics:',file=OutFile)
        print("Designated Servers: "+str(self.DesServers),file=OutFile)
        print('Total request generated = '+str(self.statistics['GeneratedRequests']),file=OutFile)
        print('Total request satisfied = '+str(self.statistics['RequestSatisfied']),file=OutFile)
        sum_computation = sum( [self.statistics['ComputePerformed'][node] for node in self.Nodes])
        print('Total computation performed = '+str(sum_computation),file=OutFile)
        print('Request delay = Min:'+str(round(self.statistics['RequestDelay']['Min'],2))\
            +', Max:'+str(round(self.statistics['RequestDelay']['Max'],2))\
            +', Average:'+str(round(self.statistics['RequestDelay']['Average'],2)),file=OutFile)
        print('Total request message handled = '+str(self.statistics['HandledRequestMessages']),file=OutFile)
        print('Total response message handled = '+str(self.statistics['HandledResponseMessages']),file=OutFile)
        sum_ServerHit = sum( [self.statistics['ServerHit'][node][item] for node in self.Nodes for item in self.Items] )
        print('Total server hit = '+str(sum_ServerHit),file=OutFile)
        sum_CacheHit = sum( [self.statistics['CacheHit'][node][item][stage] for node in self.Nodes for item in self.Items for stage in self.Stages] )
        print('Total cache hit = '+str(sum_CacheHit),file=OutFile)
        sum_CacheMiss = sum( [self.statistics['CacheMiss'][node][item][stage] for node in self.Nodes for item in self.Items for stage in self.Stages] )
        print('Total cache miss = '+str(sum_CacheMiss),file=OutFile)
        print('Total control message generated = '+str(self.statistics['GeneratedControlMessages']),file=OutFile)
        if len(self.statistics['MonitorTime']) >= 1:
            print('Total monitor times = '+str(len(self.statistics['MonitorTime'])),file=OutFile)
            print(LineBreak,file=OutFile)
            print('Monitor run time = '+str(self.statistics['MonitorRealTime']),file=OutFile)
            print(LineBreak,file=OutFile)
            print('Actual link cost = '+str(self.statistics['LinkCost_actual']),file=OutFile)
            print(LineBreak,file=OutFile)
            print('Theoretical link cost = '+str(self.statistics['LinkCost_theo']),file=OutFile)
            print(LineBreak,file=OutFile)
            print('Actual computation cost = '+str(self.statistics['CompCost_actual']),file=OutFile)
            print(LineBreak,file=OutFile)
            print('Theoretical computation cost = '+str(self.statistics['CompCost_theo']),file=OutFile)
            print(LineBreak,file=OutFile)
            print('Actual cache cost = '+str(self.statistics['CacheCost_actual']),file=OutFile)
            print(LineBreak,file=OutFile)
            print('Theoretical cache cost = '+str(self.statistics['CacheCost_theo']),file=OutFile)
            print(LineBreak,file=OutFile)
            print('Actual total cost = '+str(self.statistics['TotalCost_actual']),file=OutFile)
            print(LineBreak,file=OutFile)
            print('Theoretical total cost = '+str(self.statistics['TotalCost_theo']),file=OutFile)
            print(LineBreak,file=OutFile)
        
        #print("Node marginals: "+str( [self.Nodes[i]['InputMarg'] for i in self.Nodes ]))
    
    def ACR_solver(self,R,Paths,CacheCap,LinkWeights):
        """ solver for adaptive caching with routing, using a gradient projection. Assuming one cache type
        Input: 
        LinkWeights is a dict {edge e: weight w_e} 
        CacheCap is a dict {node v: c_v}
        R is a dict { (requester i, item k) : rate r_ik}
        Paths is a dict of list of list { (i,k) : [ list of paths starting from i] }
            note: the first path should be the shortest path"""

        MaxIter = 100
        StepSize = 0.05

        requests = R.keys()
        path_num = {}
        # start with uniform routing with no caching
        rho = {}    # path select var, dict { (i,k) : [array of rho_(i,k),p for all path p] }
        rho_grad = {}       # gradient of path selet var, same structure as rho
        for r in requests:
            path_num[r] = len(Paths[r])
            #rho[r] = np.ones(path_num[r]) / path_num[r]
            rho[r] = np.zeros(path_num[r])
            rho[r][0] = 1.0
            rho_grad[r] = np.zeros(path_num[r])
        y = {}      # caching var, dict {node i: [array of y_ik for all k]}
        y_grad = {}     # gradient of caching var, same structure
        for node in self.Nodes:
            y[node] = np.zeros(self.number_of_items)
            y_grad[node] = np.zeros(self.number_of_items)

        # iteration
        for iter in range(MaxIter):
            #print('Adaptive Cahing - Routing: iter '+str(iter))
            # reset gradients
            for r in requests:
                rho_grad[r] = np.zeros(path_num[r])
            for node in self.Nodes:
                y_grad[node] = np.zeros(self.number_of_items)
            
            # first calculate the gradient, traverse all request, path, path_pos
            for r in requests:
                item = r[1]
                for path_id in range(path_num[r]):
                    path = Paths[r][path_id]
                    for path_pos in range(1,len(path)):
                        # note : here path_pos = [1,2,|path|-1]
                        agg_var = 1.0 - rho[r][path_id]
                        for k_prime in range(path_pos):
                            # note: here k_prime = [0,1,path_pos-1]
                            p_k_prime = path[k_prime]
                            agg_var += y[p_k_prime][item]
                        if agg_var <= 1.0:
                            e = (path[path_pos],path[path_pos-1])
                            grad_inc = LinkWeights[e] * R[r]
                            rho_grad[r][path_id] -= grad_inc
                            for k_prime in range(path_pos):
                                p_k_prime = path[k_prime]
                                y_grad[p_k_prime][item] += grad_inc
            # then, update rho and y using a gradient projection
            #print('Gradient at (3,1) = '+str(y_grad[3][1]))
            sum_diff = 0.0
            # rho variables,
            for r in requests:
                rho_grad_normalized = rho_grad[r] / abs(sum(rho_grad[r])) if abs(sum(rho_grad[r])) > eps else np.ones(path_num[r]) / path_num[r]
                # note: gradient of rho is negative, will be subtracted for descend
                rho_r_temp = rho[r] + StepSize * rho_grad_normalized
                # projection
                rho_r_projected = PorjectOnSimplex(rho_r_temp)
                if abs(sum(rho_r_projected) - 1.0) > eps or min(rho_r_projected) < -eps:
                    print('ERROR: rho projection wrong, rho_r_projected = '+str(rho_r_projected))
                    exit()
                sum_diff += sum(abs(rho[r] - rho_r_projected))
                rho[r] = rho_r_projected.copy()
            # y variables
            for node in self.Nodes:
                y_grad_normalized = y_grad[node] / abs(sum(y_grad[node])) if abs(sum(y_grad[node])) > eps else np.zeros(self.number_of_items)
                # note : gradient of y is positive
                y_node_temp = y[node] + StepSize * y_grad_normalized
                # projection
                #print('y_node_temp = '+str(y_node_temp))
                y_node_projected = np.zeros(self.number_of_items)
                if CacheCap[node] == 0:
                    #y_node_projected = np.zeros(self.number_of_items)
                    pass
                elif  max(y_node_temp) <= 1.0 and sum(y_node_temp) <= CacheCap[node]:
                    y_node_projected = y_node_temp
                else:
                    #print('y_node_temp = '+str(y_node_temp))
                    y_node_projected_temp = projectToSimplex_UpperBoundBy1(y_node_temp,CacheCap[node])
                    if y_node_projected_temp is None:
                        print('ERROR: None after projection. y = '+str(y_node_temp)+' Cap = '+str(CacheCap[node]))
                        exit()
                    for item in self.Items:
                        if y_node_projected_temp[item] > 1.0- eps:
                            y_node_projected[item] = 1.0
                        elif y_node_projected_temp[item] < eps:
                            y_node_projected[item] = 0.0
                        else:
                            y_node_projected[item] = y_node_projected_temp[item]
                        #y_node_projected[item] = y_node_projected_temp[item] if y_node_projected_temp[item] > eps else 0.0
                    #print('y_node_projected = '+str(y_node_projected))
                    #print('type : '+str(type(y_node_projected)))
                #else:
                    # else, cache the top items
                #    sorted_items = sorted(range(len(y_node_temp)), key=lambda x: y_node_temp[x], reverse=True)
                #    for k in range(int(round(CacheCap[node]))):
                #        y_node_projected[sorted_items[k]] = 1.0
                
                #elif sum(y_node_temp) <= CacheCap[node]:
                #    y_node_projected = np.minimum(y_node_temp,1.0)
                #else:
                #    y_node_projected = PorjectOnSimplex(y_node_temp,CacheCap[node])
                if sum(y_node_projected) - CacheCap[node] > 0.1 or min(y_node_projected) < -eps:
                    #print('ERROR: y projection wrong, y_projected = '+str(y_node_projected))
                    print('ERROR: y projection wrong. node '+str(node)+', CacheCap = '+str(CacheCap[node])+', sum_y = '+str(sum(y_node_projected)))
                    exit()
                sum_diff += sum(abs(y[node] - y_node_projected))
                
                y[node] = y_node_projected.copy()
            if sum_diff <= eps:
                print('Stop due to sum_diff = '+str(sum_diff))
                break
        #print('y[3][1] ='+str(y[3][1])+', used cache at 3='+str(sum(y[3]))+', cache size = '+str(CacheCap[3]))
        return (rho,y)
            
    def update_process_ACR(self):
        """ Adaptive caching with routing (Stratis).
        Joint determining caching and routing, using k-shortest path """
        # construct R
        R = {}
        for node in self.Nodes:
            for item in self.Items:
                if self.RequestRates[node][item] > eps:
                    R[(node,item)] = self.RequestRates[node][item]

        # construnct Paths
        K_SP = 1
        Paths = {}
        G_temp = nx.DiGraph()
        G_temp.add_nodes_from(self.Nodes)
        for e in self.EdgesEnumerator:
            MargCost_e = self.LinkMargCal(e,0.0)
            G_temp.add_edge(e[0],e[1],weight=MargCost_e)
        for r in R:
            Paths[r] = []
            serverList = self.DesServers[r[1]]
            paths_temp = {}         # a dict {path: length} for all paths from any server to requester
            for server in serverList:
                k_sp_generator = nx.shortest_simple_paths(G_temp, server, r[0], weight='weight')
                for counter, path in enumerate(k_sp_generator):
                    #print('Path = '+str(path))
                    paths_temp[tuple(path)] = nx.path_weight(G_temp, path, weight='weight')
                    if counter == K_SP-1:
                        break
            #print('paths_temp ='+str(paths_temp))
            ShortestServerPaths_r = sorted(heapq.nlargest(K_SP, paths_temp) , key =  paths_temp.get)  # all K-SP from server to requester sorted by total weight
            #print('ShortestPaths_r = '+str(ShortestServerPaths_r))
            for server_path in ShortestServerPaths_r:
                Paths[r].append(list(server_path[::-1]))   # note: need to reverse to requester -> server

        # construct LinkWeights
        LinkWeights = {}
        for e in self.EdgesEnumerator:
            LinkWeights[e] = self.LinkMargCal(e,0.0)

        CacheCap = {}
        for node in self.Nodes:
            CacheCap[node] = 0.0

        #yield self.env.timeout(0.5)
        while True:
            #yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)
            print('AC-R updating ... ')
            # construnct CacheCap
            for node in self.Nodes:
                CacheCap[node] = sum(self.Nodes[node]['Cache'].size)

            # run AC-R
            (rho_ACR,y_ACR) = self.ACR_solver(R,Paths,CacheCap,LinkWeights)

            # update variables
            # first y
            if self.AlgoConfig.CacheTypeAlgo != 'All-first':
                print('ERROR: must use All-first in AC-R')
                exit()
            for node in self.Nodes:
                #y_temp = np.zeros(self.number_of_items)
                #sorted_items = sorted(range(len(y_ACR[node])), key=lambda x: y_ACR[node][x], reverse=True)
                #for k in range(int(round(CacheCap[node]))):
                #    if k >= len(sorted_items)-1:
                #        print('ERROR: sorted_items ='+str(sorted_items)+', k ='+str(sorted_items)+', y_temp len = '+str(len(y_temp)))
                #        exit()
                #    y_temp[sorted_items[k]] = 1.0
                for item in self.Items:
                    #self.Nodes[node]['CacheVar'][item] = np.zeros(self.CacheTypes)
                    self.Nodes[node]['CacheVar'][item][0] = y_ACR[node][item]
                    #self.Nodes[node]['CacheVar'][item][0] = y_temp[item]
                    #self.Nodes[node]['RouteVar'][item] = np.zeros(self.number_of_nodes)
            #print('y_31 = '+str(self.Nodes[3]['CacheVar'][1][0]))
            # then phi
            for r in R:
                self.Nodes[r[0]]['RouteVar'][r[1]] = np.zeros(self.number_of_nodes)
                if r[0] in self.DesServers[r[1]]:
                    continue
                #print('r = '+str(r)+' rho = '+str(rho_ACR[r]))
                for path_id in range(len(rho_ACR[r])):
                    path = Paths[r][path_id]
                    y_requester = sum(self.Nodes[r[0]]['CacheVar'][r[1]])
                    #print('r = '+str(r)+', DesServers = '+str(self.DesServers[r[1]])+', path = '+str(path)+', path_id = '+str(path_id),' + y_requester = '+str(y_requester))
                    self.Nodes[r[0]]['RouteVar'][r[1]][path[1]] = (1.0 - y_requester) * rho_ACR[r][path_id]
                    #for path_pos in range(1,len(path)-1):
                    #    node_i = path[path_pos]
                    #    node_j = path[path_pos +1]
                    #    y_node_i = sum(self.Nodes[node_i]['CacheVar'][r[1]])
                    #    self.Nodes[node_i]['RouteVar'][r[1]][node_j] = (1.0 - y_node_i)
                    #    if sum(self.Nodes[node_i]['RouteVar'][r[1]]) + y_node_i >= 1.0+ eps:
                    #        print('ERROR: sum exceeds 1: '+str(sum(self.Nodes[node_i]['RouteVar'][r[1]]) + y_node_i))
                    #        exit()
            for node in self.Nodes:
                self.optCache_mandate_shuffle(node)
            yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)
        
    def update_process_MinDelay(self):
        """ MinDelay joint caching and routing by Milad.
            Esentially a fixed-cache size version of GP.
            A result from theoretical flow amounts and using conditional Frank-Wolfe with stepsize 1.
            Each slot is an iteration, and reset to init each period. """
        # first store the init state
        RouteVar_init = {}
        CacheVar_init = {}
        for node in self.Nodes:
            RouteVar_init[node] = {}
            CacheVar_init[node] = np.zeros(self.number_of_items)
            for item in self.Items:
                RouteVar_init[node][item] = self.Nodes[node]['RouteVar'][item].copy()
                CacheVar_init[node][item] = sum(self.Nodes[node]['CacheVar'][item])
        
        if self.AlgoConfig.CacheTypeAlgo != 'All-first':
            print('Warning: MinDelay only use for All-first')

        #yield self.env.timeout(0.5)
        while True:
            #print('MinDelay: variables reset.')
            # init the variables use in one period:
            RouteVar_curr = {}
            CacheVar_curr = {}
            for node in self.Nodes:
                RouteVar_curr[node] = {}
                CacheVar_curr[node] = CacheVar_init[node].copy()
                for item in self.Items:
                    RouteVar_curr[node][item] = RouteVar_init[node][item].copy()

            for slot_id in range(self.SimConfig.L_update):
                #print('MinDelay: slot' +str(slot_id)+ ', updating...')
                # conditional gradient
                Delta,CacheScore = self.MinDelay_para_cal(RouteVar_curr,CacheVar_curr)
                for node in self.Nodes:
                    for item in self.Items:
                        RouteVar_curr[node][item] = np.zeros(self.number_of_nodes)
                        if node not in self.DesServers[item]:
                        # forward to the node with minimum delta
                            delta_vec = Delta[node][item]
                            next_hop = np.argmin(delta_vec)
                            if (node,next_hop) not in self.EdgesEnumerator:
                                print('ERROR: MinDelay forward on non-link '+str((node,next_hop))+', delta_vec = '+str(delta_vec))
                                exit()
                            RouteVar_curr[node][item][next_hop] = 1.0
                    # cache items with highest CacheScore
                    CacheCap = sum(self.Nodes[node]['Cache'].size)
                    CS_vec = CacheScore[node]
                    sorted_items = sorted(range(len(CS_vec)), key=lambda x: CS_vec[x], reverse=True)
                    CacheVar_curr[node] = np.zeros(self.number_of_items)
                    for k in range(int(round(CacheCap))):
                        CacheVar_curr[node][sorted_items[k]] = 1.0

                # update variables according to result
                for node in self.Nodes:
                    for item in self.Items:
                        self.Nodes[node]['RouteVar'][item] = (RouteVar_curr[node][item] * (1.0 - CacheVar_curr[node][item])).copy()
                        self.Nodes[node]['CacheVar'][item][0] = CacheVar_curr[node][item]
                    self.optCache_mandate_shuffle(node)
                yield self.env.timeout(self.SimConfig.T_slot)

    def MinDelay_para_cal(self,RouteVar,CacheVar):
        """ Calculate the parameters \delta_ijk and cache scores, with given RouteVar:phi and CacheVar:rho. 
        Input: RouteVar: { node i: item k: [array of phi_ijk for all j]}
            CacheVar : {node i: [array of rhi_ik for all k]}.
        Output: Dleta : { node i: item k: [array of delta_ijk for all j]}
            CacheScore : {node i: [array of CS_ik for all k]}"""

        if self.Is_looped_GivenPhi(RouteVar):
            print('ERROR: loop detected in MinDelay')
            exit()
        #RouteVar_temp = {}
        pDpr = {}
        Delta = {}  
        CacheScore = {}
        for node in self.Nodes:
            #RouteVar_temp[node] = {}
            pDpr[node] = np.ones(self.number_of_items) * np.inf
            Delta[node] = {}
            CacheScore[node] = np.zeros(self.number_of_items)
            for item in self.Items:
                #RouteVar_temp[node][item] = RouteVar[node][item].copy()
                Delta[node][item] = np.ones(self.number_of_nodes) * np.inf
        flow_theo = self.FlowCal_MinDelay(RouteVar,CacheVar)
        Flow = {}   # dict {edge e: F_e}
        Traffic = {}    # dict {node i: item k: t_ik}
        for node in self.Nodes:
            Traffic[node] = {}
            for item in self.Items:
                Traffic[node][item] = self.RequestRates[node][item]
        for e in flow_theo:
            Flow[e] = sum(flow_theo[e].values())
            for item in self.Items:
                Traffic[e[1]][item] += flow_theo[e][item]

        # first calculate partial D partial r (pDpr)
        for item in self.Items:
            # Start with servers or cache=1 nodes, iteratively check all nodes, remove the RouteVar_temp for done nodes
            undone_list = []    # nodes that have not compute pDpr
            done_list = []      # nodes that already sucessfully compute pDpr
            for node in self.Nodes:
                if node in self.DesServers[item]:# or CacheVar[node][item] >= 1.0 -eps:
                    pDpr[node][item] = 0.0
                    done_list.append(node)
                else:
                    undone_list.append(node)
            while undone_list:
                ready_list = []     # nodes will be computing pDpr at this round
                # first identify all undone nodes that have all downstream neighbors done
                for node in undone_list:
                #for node in self.Nodes:
                    RouteVar_temp = RouteVar[node][item]
                    Is_Ready = True
                    for node_j in undone_list:
                        # if there exist one undone node is a downstream neigbor, not ready.
                        if RouteVar_temp[node_j] > eps:
                            Is_Ready = False
                    if Is_Ready:
                        ready_list.append(node)
                # then calculate the pDpr of all ready node
                if not ready_list:
                    print('ERROR: ready_list empty but not all done')
                    exit()
                #print('undone_list = '+str(undone_list))
                #print('done_list = '+str(done_list))
                #print('ready_list = '+str(ready_list))
                for node in ready_list:
                    Sum = 0.0
                    for node_j in self.Nodes[node]['Neighbor']:
                        # see eq. (7)(8) in MinDelay paper
                        if RouteVar[node][item][node_j] >= eps:
                            e = (node_j,node)
                            Delta[node][item][node_j] = self.LinkMargCal(e,Flow[e]) + pDpr[node_j][item]
                            if np.isinf(Delta[node][item][node_j]):
                                print('ERROR: infinity delta')
                            #print('Delta[node][item][node_j] = '+str(self.LinkMargCal(e,Flow[e]))+'+ '+str(pDpr[node_j][item])+', RouteVar[node][item][node_j] = '+str(RouteVar[node][item][node_j]))
                            Sum += RouteVar[node][item][node_j] * Delta[node][item][node_j]
                            #Sum += RouteVar[node][item][node_j] * ( self.LinkMargCal(e,Flow[e]) + pDpr[node_j][item] )
                    pDpr[node][item] = (1 - CacheVar[node][item]) * Sum
                    if node not in undone_list:
                        print('node '+str(node)+' not in undone list')
                    undone_list.remove(node)
                    done_list.append(node)
            
        # then calculate delta from pDpr
        #for node in self.Nodes:
        #    for item in self.Items:
        #        for node_j in self.Nodes[node]['Neighbor']:
        #            Delta[node][item][node_j] = self.LinkMargCal(e,Flow[e]) + pDpr[node_j][item]

        # then calcuate CacheScores, eq. (21) in MinDelay paper
        for node in self.Nodes:
            for item in self.Items:
                delta_vec = Delta[node][item]
                if node in self.DesServers[item]:# or CacheVar[node][item] > 1.0-eps:
                    delta_ik = 0.0
                else:
                    delta_ik = min(delta_vec)
                #print('CS = '+str(Traffic[node][item])+' * '+str(delta_ik))
                if np.isinf(delta_ik):
                    print('ERROR: infinity delta_ik, delta_vec = '+str(delta_vec))
                    print('node '+str(node)+' item '+str(item)+', y_ik: '+str(sum(self.Nodes[node]['CacheVar'][item]))+', pDpr: '+str(pDpr[node][item]))
                    exit()
                CacheScore[node][item] = Traffic[node][item] * delta_ik

        return Delta,CacheScore
   
    def FlowCal_MinDelay(self,RouteVar,CacheVar):
        """ Calculate the theoretical flow from input rate r using a given Route Var andCacheVar.
        This is dedicated for Mindelay, where the sum of phi must be 1."""

        flow_theo = {}
        for e in self.EdgesEnumerator:
            flow_theo[e] = {}
            for item in self.Items:
                flow_theo[e][item] = 0.0
 
         # construct the linear equations
        n_eq = self.number_of_nodes
        for k in self.Items:
            # compute [t_i] for each item, solving linear set: A*t = b: t_i - \sum_{j in N_i} \phi_ji * t_j = r_i for all i
            length_t = self.number_of_nodes
            A = np.zeros((n_eq,length_t))
            b = np.zeros(n_eq)
            for i in self.Nodes:
            #    print('i = '+str(i)+', neighbor = '+str(self.Nodes[i]['Neighbor']))
                A[i][i] = 1.0
                for j in self.Nodes[i]['Neighbor']:
                    #A[i][j] = -1.0 * RouteVar[j][k][i] * (1.0 - CacheVar[j][k]) if RouteVar[j][k][i] > eps else 0.0
                    A[i][j] = -1.0 * RouteVar[j][k][i] if RouteVar[j][k][i] > eps else 0.0
                b[i] = self.RequestRates[i][k]
            #print('A = '+str(A))
            #print('b = '+str(b))
            f_k = np.linalg.solve(A, b)
            #print('f_k = '+str(f_k))
            for e in self.EdgesEnumerator:
            #    print('e = '+str(e[0]) + ',' +str(e[1]))
                flow_theo[e][k] = f_k[e[1]] * RouteVar[e[1]][k][e[0]]
        return flow_theo
                
    def update_process_ACN(self):
        """ Adaptive caching with network-wide cache capacity constraint (Stratis).
        Staring from 0, add total cache capacity by 1 each period. 
        Implementation is same as ACR with K_SP=1, but projection becomes Frank-Wolfe, due to the large variable dimension"""

        TotalCacheCap = 0

        R = {}
        for node in self.Nodes:
            for item in self.Items:
                if self.RequestRates[node][item] > eps:
                    R[(node,item)] = self.RequestRates[node][item]

        # construnct Paths
        Paths = {}
        G_temp = nx.DiGraph()
        G_temp.add_nodes_from(self.Nodes)
        for e in self.EdgesEnumerator:
            MargCost_e = self.LinkMargCal(e,0.0)
            G_temp.add_edge(e[0],e[1],weight=MargCost_e)
        for r in R:
            path_weight, server_path = nx.multi_source_dijkstra(G_temp,self.DesServers[r[1]],r[0],weight='weight')
            Paths[r] = list(server_path[::-1])   # note: need to reverse to requester -> server

        # construct LinkWeights
        LinkWeights = {}
        for e in self.EdgesEnumerator:
            LinkWeights[e] = self.LinkMargCal(e,0.0)

        yield self.env.timeout(0.5)
        while True:
            yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)
            print('AC-N updating ... ')
            # Add total CacheCap by 1
            TotalCacheCap += 1

            # run AC-R
            y_ACN = self.ACN_solver(R,Paths,TotalCacheCap,LinkWeights)

            # update variables
            # first y
            if self.AlgoConfig.CacheTypeAlgo != 'All-first':
                print('ERROR: must use All-first in AC-R')
                exit()
            for node in self.Nodes:
                self.Nodes[node]['Cache'].size[0] = np.ceil(sum(y_ACN[node]))
                for item in self.Items:
                    self.Nodes[node]['CacheVar'][item][0] = y_ACN[node][item]

    def update_process_ALT(self):
        """ Alternating method with Adaptive caching with network-wide cache capacity constraint (Stratis).
        Staring from 0, add total cache capacity by 1 each period. 
        Implementation is same as ACR with K_SP=1, but projection becomes Frank-Wolfe, due to the large variable dimension"""

        TotalCacheCap = 0

        R = {}
        for node in self.Nodes:
            for item in self.Items:
                if self.RequestRates[node][item] > eps:
                    R[(node,item)] = self.RequestRates[node][item]

        # construnct Paths
        Paths = {}
        G_temp = nx.DiGraph()
        G_temp.add_nodes_from(self.Nodes)
        for e in self.EdgesEnumerator:
            MargCost_e = self.LinkMargCal(e,0.0)
            G_temp.add_edge(e[0],e[1],weight=MargCost_e)
        for r in R:
            path_weight, server_path = nx.multi_source_dijkstra(G_temp,self.DesServers[r[1]],r[0],weight='weight')
            Paths[r] = list(server_path[::-1])   # note: need to reverse to requester -> server

        # construct LinkWeights
        LinkWeights = {}
        for e in self.EdgesEnumerator:
            LinkWeights[e] = self.LinkMargCal(e,0.0)

        yield self.env.timeout(0.5)
        while True:
            yield self.env.timeout(self.SimConfig.T_slot * self.SimConfig.L_update)
            print('AC-N updating ... ')
            # Add total CacheCap by 1
            TotalCacheCap += 1

            # run AC-R
            y_ACN = self.ACN_solver(R,Paths,TotalCacheCap,LinkWeights)

            # update variables
            # first y
            if self.AlgoConfig.CacheTypeAlgo != 'All-first':
                print('ERROR: must use All-first in AC-R')
                exit()
            for node in self.Nodes:
                self.Nodes[node]['Cache'].size[0] = np.ceil(sum(y_ACN[node]))
                for item in self.Items:
                    self.Nodes[node]['CacheVar'][item][0] = y_ACN[node][item]

            # then, reconstruct routing paths according to the new cache var
            # cache node list
            CacheLists = {}
            for item in self.Items:
                CacheLists[item] = self.DesServers[item]
                for node in self.Nodes:
                    if self.Nodes[node]['CacheVar'][item][0] >= 1.0 - eps and (node not in CacheLists[item]):
                        CacheLists[item].append(node)
            # then construct shortest path to any node in the cache list
            G_temp = nx.DiGraph()
            G_temp.add_nodes_from(self.Nodes)
            for e in self.EdgesEnumerator:
                MargCost_e = self.LinkMargCal(e,0.0)
                G_temp.add_edge(e[0],e[1],weight=MargCost_e)
            for r in R:
                path_weight, server_path = nx.multi_source_dijkstra(G_temp,CacheLists[r[1]],r[0],weight='weight')
                Paths[r] = list(server_path[::-1])   # note: need to reverse to requester -> server





    def ACN_solver(self,R,Paths,TotalCacheCap,LinkWeights):
        """ solver for adaptive caching with network-wide constraint, using Frank-Wolfe. Assuming one cache type
        Input: 
        LinkWeights is a dict {edge e: weight w_e} 
        CacheCap is a dict {node v: c_v}
        R is a dict { (requester i, item k) : rate r_ik}
        Paths is a dict of list { (i,k) : path from i to a server of k }"""

        MaxIter = 200
        StepSize_offset = 2     # stepsize of Frank-wolfe is set to 1/( offset + sqrt(k))

        requests = R.keys()
        y = {}      # caching var, dict {node i: [array of y_ik for all k]}
        y_grad = {}     # gradient of caching var, same structure
        for node in self.Nodes:
            y[node] = np.zeros(self.number_of_items)
            y_grad[node] = np.zeros(self.number_of_items)

        # iteration
        for iter in range(MaxIter):
            #print('Adaptive Cahing - Network-wide Constraint: iter '+str(iter))
            # reset gradients
            for node in self.Nodes:
                y_grad[node] = np.zeros(self.number_of_items)
            
            # first calculate the gradient, traverse all request, path, path_pos
            for r in requests:
                item = r[1]
                path = Paths[r]
                for path_pos in range(1,len(path)):
                    # note : here path_pos = [1,2,|path|-1]
                    agg_var = 0.0
                    for k_prime in range(path_pos):
                        # note: here k_prime = [0,1,path_pos-1]
                        p_k_prime = path[k_prime]
                        agg_var += y[p_k_prime][item]
                    if agg_var <= 1.0:
                        e = (path[path_pos],path[path_pos-1])
                        grad_inc = LinkWeights[e] * R[r]
                        for k_prime in range(path_pos):
                            p_k_prime = path[k_prime]
                            y_grad[p_k_prime][item] += grad_inc

            # then, update rho and y using a Frank-Wolfe
            # let y_LP be the linear programming solution, then y(n+1) = (1-Stepsize)y(n) + StepSize * y_LP
            Stepsize = 1/(StepSize_offset + np.sqrt(float(iter)))

            # LP solution, i.e., pick the TotalCacheCap-largest value in network-wide y_grad 
            y_LP = {}
            for node in self.Nodes:
                y_LP[node] = np.zeros(self.number_of_items)
            if TotalCacheCap >= 1:
                Grad_Heap = []       # a heap, containing smallest y_grad
                for node in self.Nodes:
                    for item in self.Items:
                        Grad_tuple = (y_grad[node][item] , (node,item))
                        #print('Grad_Heap = '+str(Grad_Heap)+', type = '+str(type(Grad_Heap)))
                        if len(Grad_Heap) < TotalCacheCap:
                            # if cache cap is not filled yet, add new item
                            heapq.heappush(Grad_Heap, Grad_tuple)
                        else:
                            # if cache cap is filled, insert only if y_grad is larger than the smallest
                            heapq.heappushpop(Grad_Heap, Grad_tuple)
                # after going through all (i,k), update y_LP
                for Grad_tuple in Grad_Heap:
                    node = Grad_tuple[1][0]
                    item = Grad_tuple[1][1]
                    y_LP[node][item] = 1.0
            
            # combining y_LP and y(n) to update y(n+1), and check availability
            sum_y = 0.0
            sum_diff = 0.0
            for node in self.Nodes:
                for item in self.Items:
                    y_new = (1-Stepsize) * y[node][item] + Stepsize * y_LP[node][item]
                    sum_diff += abs(y[node][item] - y_new)
                    y[node][item] = y_new
                sum_y += sum(y[node])
                if min(y[node]) <= -eps or max(y[node])>= 1.0+eps:
                    print('ERROR: illegal y variable at node '+str(node)+', y = '+str(y[node]))
                    exit()
            if sum_y >= TotalCacheCap + eps:
                print('ERROR: sum of y =' +str(sum_y))
                exit()
            if sum_diff <= eps:
                print('Stop due to sum_diff = '+str(sum_diff))
                break
        return y

    def Is_exceedCapacity(self):
        """ Test if the theo flow and workload of current phi will generate Flow exceeding the specified link capacity."""
        (Flow_theo, Workload_theo, traffic) = self.TheoreticalFlowCal()
        Max_exceed_amount = -np.inf     # the maximum amount of flow exceeding capacity. Will be negative if no exceeding
        Max_exceed_link = None

        for e in self.EdgesEnumerator:
            exceed_amount =  Flow_theo[e] - self.LinkCapacity[e[0]][e[1]]
            if exceed_amount > Max_exceed_amount:
                Max_exceed_amount = exceed_amount
                Max_exceed_link = e
        for i in self.Nodes:
            exceed_amount = Workload_theo[i] - self.CompCapacity[i]
            if exceed_amount > Max_exceed_amount:
                Max_exceed_amount = exceed_amount
                Max_exceed_link = i
        if Max_exceed_amount > 0.0:
            print('Max_exceed_amount = '+str(Max_exceed_amount)+', at link/node '+str(Max_exceed_link))
        else:
            print('Nearest flow to capacity at link/node '+str(Max_exceed_link)+', remaining capacity '+str(Max_exceed_amount))
        return True if Max_exceed_amount > 0.0 else False

    def Pure_Caching_Cost(self):
        """Compute the cost if each requester caches all the requested (item,stage). so that no transmission is ever needed."""
        PureCacheCost = 0.0
        for node in self.Nodes:
            CacheSize = 0.0
            for item in self.Items:
                for stage in self.Stages:
                    if self.RequestRates[node][item][stage] > eps:
                        CacheSize += 1.0
            PureCacheCost += self.CacheCostCal(node,CacheSize)
        return PureCacheCost

def main():
    "run simulator and print results"

    #testLP()
    #print(str(projectToSimplex_UpperBoundBy1(np.array([1.0,2.0,3.0]),2)))
    #exit()
    #dim = 10
    #c = 3
    #v = np.random.random(dim)
    #print('v = '+str(v))
    #print('Efficient Project: '+str(projectToSimplex_UpperBoundBy1(v,c)))
    #print('QP Project: '+str(projectToSimplex_UpperBoundBy1_legacy(v,c)))
    #exit()

    print('Simulation start...')

    parser = argparse.ArgumentParser(description = 'Simulate a Network of Elastic Cache and Service Chain Computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ConfigFile',help = \
        "Configuration file containing all other parameters. The file should be in the formate: Name Value #Description.\
        Note that when 'value' is a file name, use '' to denote the full path as a string ")

    args = parser.parse_args()
    ConfigFileName = './'+args.ConfigFile

    # Set the output file
    OutDict = './result/'  # folder for result files
    localtime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) 
    OutFileName = OutDict + 'SimOut ' + str(args.ConfigFile) + str(localtime) + '.txt'
    OutFile = open(OutFileName,'w+')
    print(LineBreak+ 'Time:\n'+ str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ), file=OutFile)

    # Read parameters from config file
    print(LineBreak+ 'Configuration:', file=OutFile)
    ConfigFile = open(ConfigFileName,'r+')
    Config_lines = ConfigFile.readlines()
    for line in Config_lines:
        line_elements = line.strip().split()
        if len(line_elements) <= 1:
            continue
        [ParaName,ParaValueStr] = line_elements[0:2]
        if ParaName.startswith('#'):
            continue
        line_exec = "%s = %s" %(ParaName,ParaValueStr)
        print(line_exec, file=OutFile)
        exec(line_exec,globals())
    #print(GraphType)
    LinkCostType_temp = LinkCostType # copy variables to avoid direct pickling parameters in file
    CompCostType_temp = CompCostType
    CacheCostType_temp = CacheCostType

    # random topology generator
    def TopologyGenerator():
        print('Generating graph...')
        G_topology = nx.DiGraph()

        if GraphType == 'Connected_ER': # connected erdos-renyi graph
            G_topology.add_nodes_from(range(N)) # nodes are labeled 0 to N-1
            G_topology.add_edges_from([(node_i, node_i+1) for node_i in range(N-1)]) # linearly connected from 0 to N-1
            G_topology.add_edges_from([(node_i+1, node_i) for node_i in range(N-1)])
            for node_i in range(N): # for every (i,j), there is edge with probability p_ER 
                for node_j in range(N - node_i):
                    if np.random.rand() < p_ER:
                        G_topology.add_edge(node_i,node_j)
                        G_topology.add_edge(node_j,node_i)
            return G_topology

        elif GraphType == 'Linear': # linearly connected  graph
            G_topology.add_nodes_from(range(N)) # nodes are labeled 0 to N-1
            G_topology.add_edges_from([(node_i, node_i+1) for node_i in range(N-1)]) # linearly connected from 0 to N-1
            G_topology.add_edges_from([(node_i+1, node_i) for node_i in range(N-1)])
            return G_topology
        
        elif GraphType == 'grid': # 2d-grid network,
            L_grid = int(np.floor(np.sqrt(N)))
            for row in range(L_grid):
                for column in range(L_grid):
                    node_id = row * L_grid + column
                    #print('adding node '+str(node_id))
                    G_topology.add_node(node_id)
                    # connect to the left node
                    if column != 0:
                        G_topology.add_edge(node_id,node_id-1)
                        G_topology.add_edge(node_id-1,node_id)
                    # connect to the above node
                    if row != 0:
                        G_topology.add_edge(node_id,node_id-L_grid)
                        G_topology.add_edge(node_id-L_grid,node_id)
            return G_topology


        elif GraphType == 'Tree': # a complete tree, note that if N is not a node number of a complete tree, the graph uses a BFS-fill up.
            tree_deg = deg_TREE # number of children of each node
            G_topology.add_nodes_from(range(N))
            total_edge = N-1
            placed_edge = 0
            #current_node = 0
            for current_node in range(N):
                for child_id in range(tree_deg):
                    child = current_node * tree_deg + child_id + 1
                    G_topology.add_edge(current_node,child)
                    G_topology.add_edge(child,current_node)
                    print('adding edge '+str((current_node,child)))
                    placed_edge += 1
                    if placed_edge >= total_edge:
                        return G_topology

        elif GraphType == 'Fog': # a complete tree where children of the same parent are joined by aditional linear links.
            tree_deg = deg_TREE # number of children of each node
            G_topology.add_nodes_from(range(N))
            #current_node = 0
            for current_node in range(N):
                for child_id in range(tree_deg):
                    child = current_node * tree_deg + child_id + 1
                    if child >= N:
                        return G_topology
                    G_topology.add_edge(current_node,child)
                    G_topology.add_edge(child,current_node)
                    if child_id != 0:
                        G_topology.add_edge(child,child-1)
                        G_topology.add_edge(child-1,child)
            return G_topology
        
        elif GraphType == 'LHC':   # pre-defined topology for LHC
            NodeName = ['S1','S2','S3','S4','S5','S6','S7','S8','NBR','UCSD','FNL','VND','UFL','WSC','MIT','PRD']
            NameIDmap = dict(zip(NodeName,range(len(NodeName))))
            G_topology.add_nodes_from(NameIDmap.values())

            Edges = [ ('NBR','S6'), ('NBR','FNL'), ('NBR','S4'), ('UCSD','S6'), ('UCSD','FNL'), ('UCSD','S8'), ('FNL','S6'), ('FNL','S8'), ('FNL','S1'), ('VND','S6'), ('VND','S3'),('VND','UFL'), ('UFL','S5'), ('UFL','S7'), ('UFL','S3'), \
                ('WSC','S6'), ('WSC','S1'), ('MIT','S6'), ('MIT','S1'), ('MIT','S5'), ('PRD','S2'), ('S1','S2'), ('S1','S5'), ('S2','S3'), ('S2','S4'), ('S4','S5'), ('S4','S6'), ('S5','S6'), ('S5','S7'), ('S6','S8'), ('S7','S8') ]
            EdgesID = [(NameIDmap[e[0]],NameIDmap[e[1]]) for e in Edges]
            EdgesID_reverse = [(NameIDmap[e[1]],NameIDmap[e[0]]) for e in Edges]
            G_topology.add_edges_from(EdgesID)
            G_topology.add_edges_from(EdgesID_reverse)
            return G_topology

        elif GraphType == 'GEANT':   # pre-defined topology for GEANT
            G_topology.add_nodes_from(range(22))
            Edges = [(1,2),(1,22),(2,3),(2,4),(2,16),(3,4),(4,13),(4,5),(5,22),(5,19),(5,8),(6,7),(6,15),(7,8),(8,9),(9,10),(10,11),(10,13),(11,12),(12,13),\
                (13,14),(13,16),(14,15),(15,16),(15,17),(15,22),(16,18),(17,18),(18,19),(19,20),(19,21),(20,21),(21,22)]
            Edges_modified = [(e[0]-1,e[1]-1) for e in Edges]
            Edges_reverse = [(e[1]-1,e[0]-1) for e in Edges]
            G_topology.add_edges_from(Edges_modified)
            G_topology.add_edges_from(Edges_reverse)
            return G_topology

        elif GraphType == 'Dtelekom':   # pre-defined topology for Dtelekom
            G_topology.add_nodes_from(range(68))

            G_topology.add_edge(67,52)
            G_topology.add_edge(67,4)
            G_topology.add_edge(66,5)
            G_topology.add_edge(66,4)
            G_topology.add_edge(65,62)
            G_topology.add_edge(65,4)
            G_topology.add_edge(65,26)
            G_topology.add_edge(64,4)
            G_topology.add_edge(64,23)
            G_topology.add_edge(63,45)
            G_topology.add_edge(63,51)
            G_topology.add_edge(62,23)
            G_topology.add_edge(62,28)
            G_topology.add_edge(61,5)
            G_topology.add_edge(61,4)
            G_topology.add_edge(61,23)
            G_topology.add_edge(61,45)
            G_topology.add_edge(60,3)
            G_topology.add_edge(60,4)
            G_topology.add_edge(60,5)
            G_topology.add_edge(60,17)
            G_topology.add_edge(59,38)
            G_topology.add_edge(59,37)
            G_topology.add_edge(59,47)
            G_topology.add_edge(59,42)
            G_topology.add_edge(59,58)
            G_topology.add_edge(59,45)
            G_topology.add_edge(59,40)
            G_topology.add_edge(59,23)
            G_topology.add_edge(59,28)
            G_topology.add_edge(59,27)
            G_topology.add_edge(59,9)
            G_topology.add_edge(58,45)
            G_topology.add_edge(58,5)
            G_topology.add_edge(58,4)
            G_topology.add_edge(58,3)
            G_topology.add_edge(58,1)
            G_topology.add_edge(57,5)
            G_topology.add_edge(57,4)
            G_topology.add_edge(57,2)
            G_topology.add_edge(56,23)
            G_topology.add_edge(56,9)
            G_topology.add_edge(56,6)
            G_topology.add_edge(56,5)
            G_topology.add_edge(56,4)
            G_topology.add_edge(56,3)
            G_topology.add_edge(56,1)
            G_topology.add_edge(55,40)
            G_topology.add_edge(55,9)
            G_topology.add_edge(55,8)
            G_topology.add_edge(55,6)
            G_topology.add_edge(55,5)
            G_topology.add_edge(55,3)
            G_topology.add_edge(55,1)
            G_topology.add_edge(54,42)
            G_topology.add_edge(54,6)
            G_topology.add_edge(54,5)
            G_topology.add_edge(54,4)
            G_topology.add_edge(54,3)
            G_topology.add_edge(54,1)
            G_topology.add_edge(53,51)
            G_topology.add_edge(53,45)
            G_topology.add_edge(53,23)
            G_topology.add_edge(53,9)
            G_topology.add_edge(53,5)
            G_topology.add_edge(52,9)
            G_topology.add_edge(52,5)
            G_topology.add_edge(52,4)
            G_topology.add_edge(52,3)
            G_topology.add_edge(52,1)
            G_topology.add_edge(51,48)
            G_topology.add_edge(51,45)
            G_topology.add_edge(51,27)
            G_topology.add_edge(51,23)
            G_topology.add_edge(51,5)
            G_topology.add_edge(50,5)
            G_topology.add_edge(50,4)
            G_topology.add_edge(50,3)
            G_topology.add_edge(50,2)
            G_topology.add_edge(50,1)
            G_topology.add_edge(49,23)
            G_topology.add_edge(49,5)
            G_topology.add_edge(49,4)
            G_topology.add_edge(49,3)
            G_topology.add_edge(49,1)
            G_topology.add_edge(48,5)
            G_topology.add_edge(48,4)
            G_topology.add_edge(48,3)
            G_topology.add_edge(48,1)
            G_topology.add_edge(47,6)
            G_topology.add_edge(47,1)
            G_topology.add_edge(46,40)
            G_topology.add_edge(46,5)
            G_topology.add_edge(45,48)
            G_topology.add_edge(45,4)
            G_topology.add_edge(45,3)
            G_topology.add_edge(45,1)
            G_topology.add_edge(44,6)
            G_topology.add_edge(44,4)
            G_topology.add_edge(44,3)
            G_topology.add_edge(44,2)
            G_topology.add_edge(44,1)
            G_topology.add_edge(43,26)
            G_topology.add_edge(43,10)
            G_topology.add_edge(43,6)
            G_topology.add_edge(43,5)
            G_topology.add_edge(43,4)
            G_topology.add_edge(43,1)
            G_topology.add_edge(42,40)
            G_topology.add_edge(42,5)
            G_topology.add_edge(42,4)
            G_topology.add_edge(42,3)
            G_topology.add_edge(42,1)
            G_topology.add_edge(41,31)
            G_topology.add_edge(41,5)
            G_topology.add_edge(41,4)
            G_topology.add_edge(41,3)
            G_topology.add_edge(41,1)
            G_topology.add_edge(40,23)
            G_topology.add_edge(40,9)
            G_topology.add_edge(40,8)
            G_topology.add_edge(40,4)
            G_topology.add_edge(40,3)
            G_topology.add_edge(40,1)
            G_topology.add_edge(39,19)
            G_topology.add_edge(39,17)
            G_topology.add_edge(39,6)
            G_topology.add_edge(39,1)
            G_topology.add_edge(38,23)
            G_topology.add_edge(38,6)
            G_topology.add_edge(38,5)
            G_topology.add_edge(38,4)
            G_topology.add_edge(38,1)
            G_topology.add_edge(37,23)
            G_topology.add_edge(36,6)
            G_topology.add_edge(36,5)
            G_topology.add_edge(36,3)
            G_topology.add_edge(36,1)
            G_topology.add_edge(35,5)
            G_topology.add_edge(35,4)
            G_topology.add_edge(35,3)
            G_topology.add_edge(35,1)
            G_topology.add_edge(34,26)
            G_topology.add_edge(34,23)
            G_topology.add_edge(34,5)
            G_topology.add_edge(34,4)
            G_topology.add_edge(34,1)
            G_topology.add_edge(33,23)
            G_topology.add_edge(33,18)
            G_topology.add_edge(33,15)
            G_topology.add_edge(33,9)
            G_topology.add_edge(33,3)
            G_topology.add_edge(32,6)
            G_topology.add_edge(32,5)
            G_topology.add_edge(32,4)
            G_topology.add_edge(32,3)
            G_topology.add_edge(32,2)
            G_topology.add_edge(32,1)
            G_topology.add_edge(31,4)
            G_topology.add_edge(31,3)
            G_topology.add_edge(30,6)
            G_topology.add_edge(30,5)
            G_topology.add_edge(30,4)
            G_topology.add_edge(30,3)
            G_topology.add_edge(30,1)
            G_topology.add_edge(29,26)
            G_topology.add_edge(29,6)
            G_topology.add_edge(29,5)
            G_topology.add_edge(29,4)
            G_topology.add_edge(29,3)
            G_topology.add_edge(29,1)
            G_topology.add_edge(28,6)
            G_topology.add_edge(28,5)
            G_topology.add_edge(28,4)
            G_topology.add_edge(28,3)
            G_topology.add_edge(28,1)
            G_topology.add_edge(27,5)
            G_topology.add_edge(27,3)
            G_topology.add_edge(26,12)
            G_topology.add_edge(26,6)
            G_topology.add_edge(26,5)
            G_topology.add_edge(26,2)
            G_topology.add_edge(25,6)
            G_topology.add_edge(25,5)
            G_topology.add_edge(25,1)
            G_topology.add_edge(24,23)
            G_topology.add_edge(24,6)
            G_topology.add_edge(24,5)
            G_topology.add_edge(24,4)
            G_topology.add_edge(24,3)
            G_topology.add_edge(24,1)
            G_topology.add_edge(23,6)
            G_topology.add_edge(23,5)
            G_topology.add_edge(22,6)
            G_topology.add_edge(22,5)
            G_topology.add_edge(21,6)
            G_topology.add_edge(21,5)
            G_topology.add_edge(21,4)
            G_topology.add_edge(21,3)
            G_topology.add_edge(21,1)
            G_topology.add_edge(20,6)
            G_topology.add_edge(20,5)
            G_topology.add_edge(20,4)
            G_topology.add_edge(20,3)
            G_topology.add_edge(20,2)
            G_topology.add_edge(20,1)
            G_topology.add_edge(19,17)
            G_topology.add_edge(19,6)
            G_topology.add_edge(19,5)
            G_topology.add_edge(18,4)
            G_topology.add_edge(18,3)
            G_topology.add_edge(18,1)
            G_topology.add_edge(17,1)
            G_topology.add_edge(16,5)
            G_topology.add_edge(16,4)
            G_topology.add_edge(16,3)
            G_topology.add_edge(16,1)
            G_topology.add_edge(15,11)
            G_topology.add_edge(15,9)
            G_topology.add_edge(15,6)
            G_topology.add_edge(15,5)
            G_topology.add_edge(15,4)
            G_topology.add_edge(14,4)
            G_topology.add_edge(14,3)
            G_topology.add_edge(14,1)
            G_topology.add_edge(13,6)
            G_topology.add_edge(13,4)
            G_topology.add_edge(13,3)
            G_topology.add_edge(13,2)
            G_topology.add_edge(12,5)
            G_topology.add_edge(12,4)
            G_topology.add_edge(11,6)
            G_topology.add_edge(11,5)
            G_topology.add_edge(11,4)
            G_topology.add_edge(11,2)
            G_topology.add_edge(11,1)
            G_topology.add_edge(10,6)
            G_topology.add_edge(10,4)
            G_topology.add_edge(10,3)
            G_topology.add_edge(10,1)
            G_topology.add_edge(9,6)
            G_topology.add_edge(9,5)
            G_topology.add_edge(9,4)
            G_topology.add_edge(9,3)
            G_topology.add_edge(9,1)
            G_topology.add_edge(8,6)
            G_topology.add_edge(8,4)
            G_topology.add_edge(8,3)
            G_topology.add_edge(7,6)
            G_topology.add_edge(7,5)
            G_topology.add_edge(7,4)
            G_topology.add_edge(7,3)
            G_topology.add_edge(7,1)
            G_topology.add_edge(6,5)
            G_topology.add_edge(6,4)
            G_topology.add_edge(6,3)
            G_topology.add_edge(6,1)
            G_topology.add_edge(5,4)
            G_topology.add_edge(5,3)
            G_topology.add_edge(5,2)
            G_topology.add_edge(5,1)
            G_topology.add_edge(4,3)
            G_topology.add_edge(4,2)
            G_topology.add_edge(4,1)
            G_topology.add_edge(3,2)
            G_topology.add_edge(3,1)
            G_topology.add_edge(2,1)
            G_topology.add_edge(0,1)
            G_topology.add_edge(0,2)
            G_topology.add_edge(0,3)
            G_topology.add_edge(0,4)
            G_topology.add_edge(0,5)
            G_topology.add_edge(0,6)
            Edges = list(G_topology.edges)
            #print('Edges = '+str(Edges))
            Edges_reverse = [(e[1],e[0]) for e in Edges]
            #print('Edges_reverse = '+str(Edges_reverse))
            G_topology.add_edges_from(Edges_reverse)
            return G_topology

        elif GraphType == 'Small-World':    # small world topology 
            G_o = nx.connected_watts_strogatz_graph(n = N, k = 6, p = 0.1)
            Nodes = list(G_o.nodes())
            Edges = list(G_o.edges())
            Edges_reverse = [(e[1],e[0]) for e in Edges]
            G_topology.add_nodes_from(Nodes)
            G_topology.add_edges_from(Edges)
            G_topology.add_edges_from(Edges_reverse)
            return G_topology
        
        else:
            print('ERROR: Undefined graph type.')
            exit()
            #return G_topology
    
    # Create directed graph with unit link weight (will be overwrite if IsReadScenario == True)
    G = TopologyGenerator()
    print('Done')
    nx.draw(G)
    plt.savefig("testGraph.png")
    N_actual = G.number_of_nodes()
    E_actual = G.number_of_edges()
    print('Graph generated: '+str(N_actual)+' nodes, '+str(E_actual)+' edges.')

    # Generate Application-Task table
    if LenType == 'Max':
        AppChain_Len = [Len]*A  # all applications have chain length Len
    elif LenType == 'Random':
        AppChain_Len = [random.choice(range(Len + 1)) for a in range(A)]  # random chain length from 0 to Len 
    else:
        print('ERROR: Undefined LenType')

    # Generate packet sizes
    PacketSize = {}     #  PacketSize is a nested-dict, {item: { stage: size}}
    for item in range(K):
        if ItemSizeType == 'Uniform':
            item_size = random.uniform(ItemSizeMin,ItemSizeMax) 
            PacketSize[item] = {}
        else:
            print('ERROR: Undefined packet size type')
            exit()
        for app in range(A):
            for task in range(AppChain_Len[app]+1):   # from 0 to len
                stage = (app,task)
                PacketSize[item][stage] = item_size  * (StageShrink ** task)
               

    # Generate computation workloads, separate for each (app,task#).
    # task number starts from 0 to len.
    # Note: workload is independent from packetsize, workload is also assigned for stage 0, but will not be used
    CompWorkload = {}   # stage: workload
    for app in range(A):
        for task in range(AppChain_Len[app]+1):
            stage = (app,task)
            #if task != 0:
            CompWorkload[stage] = random.uniform(WorkloadMin ,WorkloadMax)

    # Assign link cost parameters for each link, uniform from LinkParaMin to LinkParaMax
    for e in G.edges():
        G[e[0]][e[1]]['LinkPara'] = random.uniform(LinkParaMin,LinkParaMax)

    # Assign computation cost parameter for each node, uniform (note: this is only the comp cost function, the comp workload is defined separately)
    for n in range(N_actual):
        G.nodes[n]['CompPara'] = random.uniform(CompParaMin ,CompParaMax)

    # Assign cache cost parameters for each node, uniform from CacheParaMin to CacheParaMax
    for n in range(N_actual):
        G.nodes[n]['CachePara'] = random.uniform(CacheParaMin,CacheParaMax)

    # Determine the designated servers
    if ServerDistribution == 'Single': # for single server, all items have the same server : node 0
        DesServerList = [0]*K 
    elif ServerDistribution == 'Uniform': # uniformly randomly choosen server
        DesServerList = [random.choice(range(N)) for k in range(K)]
    #print(DesServerList)

    # Generates demands. Each demand is a tuple (id, requester, item, stage, rate), where stage itself is a tuple (app,task). 
    # Here ``task'' can be 0 to chain length to incorporate data/result
    Demands = []
    for d_id in range(R):
        if RequetserDistribution == 'Uniform': # if the requester is uniformly distributed
            requester = random.choice(range(N_actual))

        if ItemDistribution == 'Uniform': # if the item is uniformly distributed
            item = random.choice(range(K))
        elif ItemDistribution == 'Zipf': # if the item is Zipf distributed
            item = Zipf(ItemDistPara,1,K)-1

        if AppDistribution == 'Uniform':    # if the target application is uniform
            app = random.choice(range(A))

        if IntermediateRequest is True:     # if request for intermediate results
            task = random.choice(range(AppChain_Len[app]+1))   # from 0 (data) to len (final result)
        else:
            task = AppChain_Len[app]                        # only request for final result

        stage = (app,task)
        rate =  random.uniform(RateMin,RateMax)
        Demands.append( (d_id, requester, item, stage, rate) )

    # Load scenario
    if IsReadScenario:
        ReadScenarioFile = open(ReadScenarioName,'rb')
        [G,AppChain_Len,PacketSize,CompWorkload,DesServerList,Demands,LinkCostType_temp,CompCostType_temp,CacheCostType_temp] = pickle.load(ReadScenarioFile)
        ReadScenarioFile.close()

    # test
    print('N = '+str(G.number_of_nodes()))
    for n in range(G.number_of_nodes()):
        #G.nodes[n]['CachePara'] = [random.uniform(4.0,4.01) for m in range(len(CacheCostType_temp))]
        pass

    # Save scenario
    if IsWriteScenario:
        if os.path.exists("demofile.txt"):
            os.remove(WriteScenarioName)
        WriteScenarioFile = open(WriteScenarioName,'wb')
        pickle.dump([G,AppChain_Len,PacketSize,CompWorkload,DesServerList,Demands,LinkCostType_temp,CompCostType_temp,CacheCostType_temp], WriteScenarioFile)
        WriteScenarioFile.close()

    # Simulation setting, will not effect scenario details
    SimConfig = SimulationConfig(T_sim, T_slot, L_update, T_monitor, Is_draw_network, MonitorInitTime,T_IDReset,T_extension,\
        RequestGenerateType, RequestDelayType, RequestDelay, RequestSize,\
        ResponseDelayType, ResponseDelay, ControlDelay, ControlSize)

    # Algorithm setting
    AlgoConfig = AlgorithmConfig(CacheAlgo,RouteAlgo,SizeAlgo,StepsizeGP_phi,StepsizeGP_y,Thrt,N_GCFW,RouteInitType,\
        AllowLoop,RouteType)

    # Construct ElasticCacheNetwork 
    CacheCompNet = ElasticCacheComputeNetwork(G,AppChain_Len,PacketSize,CompWorkload,DesServerList,Demands,LinkCostType_temp,CompCostType_temp,CacheCostType_temp,SimConfig,AlgoConfig)

    # Run simulation and print statistics
    t_start = time.time()
    CacheCompNet.runSim()
    t_finish = time.time()
    CPU_time = t_finish- t_start
    CacheCompNet.printStatistics()
    CacheCompNet.printStatistics(OutFile)

    #print(nx.number_of_nodes(G))

    # close files and finish
    print("total CPU time: "+str(CPU_time)+"\n",file=OutFile)
    print(LineBreak, file=OutFile)
    OutFile.close()
    ConfigFile.close()
    print('Simulation accomplished.')

if __name__=="__main__":
    main()

