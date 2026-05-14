import torch
from torch_geometric.data import Data
import torch_geometric.nn.functional as F_geom
from torch_geometric.nn import GCNConv, MLP, MessagePassing, knn_graph, radius_graph, NNConv
from torch_scatter import scatter_add
from torch_geometric.nn import pool
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import softmax
from torch.nn import ModuleList
import torch.nn as nn
import os
import glob
import math
from typing import Optional, Tuple, Dict
import torch.nn.functional as F
from models.autoencoder_model import ImageGraphAE

# Data format:
# Input:
# latent    [N,latent_dim]
# large pcd [x,y,z,R,G,B] (coarsened, partial)
# Output:
# transform [tx,ty,tz]
# view dir  [u,v,w]

device = torch.device('cuda')
ae = ImageGraphAE(latent_dim=128,use_z_norm=False,use_ss=False).to(device)

class Locator(nn.Module):
    def __init__(self,nodes_dim:int,latent_dim:int,nodes_k:int,nodes_k_pcd:int,layers:int,layers_mlp:int):
        super().__init__()
        self.latent_dim = latent_dim
        self.nodes_k = nodes_k # number of neighbors for each latent node
        self.nodes_k_pcd = nodes_k_pcd
        self.layers = layers
        self.nodes_dim = nodes_dim
        self.layers_mlp = layers_mlp

        # pcd transform: nodes from [x,y,z,R,G,B] --> 128
        self.pcd_mlp = MLP(in_channels=6,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=layers_mlp,act='relu',norm='layer')

        # latent graph transform (pcd): nodes from 256 --> 128
        self.latent_conv = ModuleList([GCNConv(in_channels=latent_dim*2,out_channels=latent_dim)
                                       for _ in range(layers)])
        
        # latent graph (observations) transform: nodes from 256 --> 128
        self.obs_conv = GCNConv(in_channels=latent_dim*2,out_channels=latent_dim)

        # node generator latent --> N nodes
        #self.node_mlp = nn.Sequential(nn.Linear(latent_dim, latent_dim),
        #    nn.ReLU(),nn.Linear(latent_dim, latent_dim),
        #    nn.ReLU(),nn.Linear(latent_dim, self.nodes_init * self.nodes_dim))
        
        # N nodes --> latent
        in_edge = 4 * nodes_dim
        self.edge_mlp = nn.Sequential(nn.Linear(in_edge, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1))

        # prediction head MLP: 128 --> [tx,ty,tz,u,v,w]
        self.translation_projection = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim,latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim,6))

        # prediction head MLP: 3*128 --> [tx,ty,tz]
        self.translation_projection_2 = nn.Sequential(
            nn.Linear(3*latent_dim,2*latent_dim),
            nn.Tanh(),
            nn.Linear(2*latent_dim,latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim,3))
        
    
    def connect_graphs(g1, g2) -> torch.Tensor:
        # take latent graph and lifted point cloud graph
        # generate edges from each vantage point to the large graph
        # losses: compare each point's x,y,z,R,G,B to correct alignment within point cloud
        # ==> view direction is implicit, calculate later from edges
        
        return

    def construct_graph(self, x: torch.Tensor, k: int):
        """
        x: [N,F] for a single graph
        Returns edge_index [2,E]
        """
        # edges via kNN in feature space
        edge_index = knn_graph(x, k=k, loop=False)  # [2,E]

        # Add reverse edges (simple symmetrization)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        src,dst = edge_index
        edge_weights = (x[src] - x[dst]).norm(p=2, dim=1, keepdim=True)
        edge_attr = x[src] - x[dst]

        return edge_index, edge_weights, edge_attr

    def self_supervision():
        # TODO

        # inspired by SeedGNN, see paper at: https://arxiv.org/pdf/2205.13679
        # use already discovered connections to inform new connections
        return
    
    def forward(self,graphs_pcd_batch, pcd_ei, pcd_ea, graph_latent_batch):

        # construct knn graph from latents
        ## latent graph node <-128d edges-> latent graph node --calculated
        #print(graph_latent_batch)
        graph_latent = graph_latent_batch['latent'] #graph_latent_batch.batch
        ei_latent, ew_latent, ea_latent = self.construct_graph(graph_latent,self.nodes_k)

        # construct latent graph from PCD graph (by default high connectivity)
        # pcd graphs are already partitioned by cluster
        # convert all clusters into latent representation and get aggregated mean by cluster
        ###
        #graph_pcd, ei_pcd, ea_pcd, b = graphs_pcd_batch.x.float(), pcd_ei, pcd_ea, graphs_pcd_batch.batch
        #x = self.pcd_mlp(graph_pcd)
        #for conv in self.latent_conv:
        #    x_global = pool.global_mean_pool(x, b)
        #    x_expanded = x_global[b]
        #    m = conv(torch.cat([x, x_expanded], dim=1), ei_pcd, edge_weight=ea_pcd)
        #    x = x+m
        #g = pool.global_mean_pool(x,b) # latent pcd graph
        ###

        ## latent graph node pose <-edges in xyz-> pcd cluster mean xyz --calculated, loss function
        # similarity metric
        # invisible/far-away subgraphs should return zero
        
        # run conv through latent graph
        #print(graph_latent.shape)
        latent_global = graph_latent.mean(dim=0,keepdim=True)
        latent_global = latent_global.expand(graph_latent.size(0), -1) 
        gl_exp = latent_global
        lat_m = self.obs_conv(torch.cat([graph_latent,gl_exp],dim=1),ei_latent,edge_weight=ew_latent) # latent message

        # TODO: add graph of pcd graphs for localization
        # TODO: find matching subgraphs based on actual pose
        # TODO: find matching subgraphs based on cosine similarity
        # TODO: use entire point cloud to calculate distances
        # graph to latent space generator from autoencoder
        g = ae.pcd_enc(graphs_pcd_batch.x.float(),graphs_pcd_batch.batch,pcd_ei,pcd_ea.float()) #alternative using pcd encoder from autoencoder
        # pose relation based on distances

        # concatenate [g, lat_m, |g-lat_m|] --> MLP
        g_exp = g[:,None,:]
        lm_exp = lat_m[None,:,:]
        exp_concat = torch.cat([g_exp.expand(-1,lat_m.size(0),-1),lm_exp.expand(g.size(0),-1,-1),
                         torch.abs(g_exp-lm_exp)],dim=-1).reshape(-1,self.latent_dim*3)
        
        # similarity score ==> gate
        # use single point autoencoder for generating latent representation of subgraphs
        # calculate similarity instead of distance  ==> extrapolate pose based on admissible samples
        similarity = F.cosine_similarity(exp_concat[:,:self.latent_dim],exp_concat[:,self.latent_dim:2*self.latent_dim],dim=1)
        gate = torch.sigmoid((similarity-0.4)/0.1).unsqueeze(-1)
        exp_concat_gated = exp_concat*gate

        predicted_distances = self.translation_projection_2(exp_concat_gated)

        #dist_node_cluster = torch.cdist(g,lat_m) # [N_pcd,N_obs,128]
        #print(g.shape,lat_m.shape,dist_node_cluster.shape)
        #predicted_distances = self.translation_projection(dist_node_cluster)
        predicted_transforms = self.translation_projection(ea_latent) # node -> node transform

        ## simultaneously: learn translations between vantage points - based on graph
        

        ## latent graph node <-128d edges-> pcd cluster mean (latent) ==> translation.py

        # outputs: 
        # predicted_distances ==> distance of given latent node to given cluster
        # predicted_transforms ==> inter-node transforms in camera basis
        # ei_latent ==> edge indices of observation graph
        return predicted_distances, predicted_transforms, ei_latent # [N_pcd,N_obs,1] -transform,-view_dir-