import pandas as pd
import igraph
import leidenalg as la
from typing import Any,Iterable,List
import logging
import numpy as np
from copy import deepcopy
from sklearn.metrics import auc

def load_network(edgelist: Iterable[Iterable[Any]]) -> igraph.Graph:
    
    g = igraph.Graph.TupleList(edgelist)
    components = sorted(g.components(),key=len,reverse=True)
    if len(components)>1:
        logging.getLogger().warn(f'Passed network has {len(components)} components. Selecting the largest one with {len(components[0])} nodes')
        g = g.subgraph(components[0])
        
    return g


def dismantle_community(g: igraph.Graph,
                        community_detection_algo: str = 'leiden',
                        extract_progress_dismantling: bool = True,
                        stop_dismantle: int = 3) -> List[Any]:
    
    g = deepcopy(g)
    removal_ranking = []
    if extract_progress_dismantling:
        n_nodes = g.vcount()
        step = 0
    while g.vcount()>stop_dismantle:

        names = g.vs['name']
        name_dict = dict(zip(g.vs['name'],range(g.vcount())))
        
        if community_detection_algo=='leiden':
            comms = list(la.find_partition(g, la.ModularityVertexPartition))
        elif community_detection_algo=='louvain':
            comms = list(g.community_multilevel())
        elif community_detection_algo=='infomap':
            comms = list(g.community_infomap())
        else:
            raise ValueError(f'Value "{community_detection_algo}" for community detection algorithm is not valid. Accepted values are "leiden", "louvain" or "infomap"')
    
        deg = dict(zip(g.vs['name'],g.degree()))
        deg_in = {}
        for com in comms:
            sub = g.subgraph(com)
            deg_in.update(dict(zip(sub.vs['name'],sub.degree())))

        to_remove = sorted([(i,deg[i],deg_in[i],(deg[i]-deg_in[i])) for i in deg],key=lambda x: (x[3],x[1],x[2]),reverse=True)[0][0]
        g.delete_vertices(name_dict[to_remove])
        if extract_progress_dismantling:
            step+=1
            components = sorted(map(len,g.components()),reverse=True)
            fc = components[0]
            if len(components)>1:
                sc = components[1]
            else: sc = 0
            removal_ranking.append((to_remove,step/n_nodes,fc,sc))
        else:
            removal_ranking.append(to_remove)
        
    return removal_ranking

def dismantle_degree(g: igraph.Graph,
                    extract_progress_dismantling: bool = True,
                    stop_dismantle: int = 3):
    
    g = deepcopy(g)
    removal_ranking = []
    if extract_progress_dismantling:
        n_nodes = g.vcount()
        step = 0
        
    while g.vcount()>stop_dismantle:

        names = g.vs['name']
        name_dict = dict(zip(g.vs['name'],range(g.vcount())))
        deg = dict(zip(g.vs['name'],g.degree()))
        to_remove = sorted([(i,deg[i]) for i in deg],key=lambda x: (x[1]),reverse=True)[0][0]
        g.delete_vertices(name_dict[to_remove])
        if extract_progress_dismantling:
            step+=1
            components = sorted(map(len,g.components()),reverse=True)
            fc = components[0]
            if len(components)>1:
                sc = components[1]
            else: sc = 0
            removal_ranking.append((to_remove,step/n_nodes,fc,sc))
        else:
            removal_ranking.append(to_remove)
        
    return removal_ranking

def dismantle_betweenness(g: igraph.Graph,
                        extract_progress_dismantling: bool = True,
                        stop_dismantle: int = 3):
    
    g = deepcopy(g)
    removal_ranking = []
    if extract_progress_dismantling:
        n_nodes = g.vcount()
        step = 0
    
    while g.vcount()>stop_dismantle:

        names = g.vs['name']
        name_dict = dict(zip(g.vs['name'],range(g.vcount())))
        bet = dict(zip(g.vs['name'],g.betweenness()))
        to_remove = sorted([(i,bet[i]) for i in bet],key=lambda x: (x[1]),reverse=True)[0][0]
        g.delete_vertices(name_dict[to_remove])
        if extract_progress_dismantling:
            step+=1
            components = sorted(map(len,g.components()),reverse=True)
            fc = components[0]
            if len(components)>1:
                sc = components[1]
            else: sc = 0
            removal_ranking.append((to_remove,step/n_nodes,fc,sc))
        else:
            removal_ranking.append(to_remove)
        
    return removal_ranking

def dismantle_ci(g: igraph.Graph,
                l: int =2,
                extract_progress_dismantling: bool = True,
                stop_dismantle: int = 3):
    
    g = deepcopy(g)
    removal_ranking = []
    if extract_progress_dismantling:
        n_nodes = g.vcount()
        step = 0
    
    while g.vcount()>stop_dismantle:

        names = g.vs['name']
        name_dict = dict(zip(g.vs['name'],range(g.vcount())))

        degree = dict(zip(range(g.vcount()),g.degree()))
        neighs = dict(zip(range(g.vcount()),g.neighborhood(mindist=l,order=l)))

        ci = {names[i]: (degree[i]-1)*sum(degree[j]-1 for j in neighs[i]) for i in degree}

        
        to_remove = sorted([(i,ci[i]) for i in names],key=lambda x: (x[1]),reverse=True)[0][0]
        g.delete_vertices(name_dict[to_remove])
        if extract_progress_dismantling:
            step+=1
            components = sorted(map(len,g.components()),reverse=True)
            fc = components[0]
            if len(components)>1:
                sc = components[1]
            else: sc = 0
            removal_ranking.append((to_remove,step/n_nodes,fc,sc))
        else:
            removal_ranking.append(to_remove)
        
    return removal_ranking            

def compute_r(fraction_removed,size_first_component):
    size_first_component = np.array(size_first_component)/max(size_first_component)
    return auc(fraction_removed,size_first_component)
