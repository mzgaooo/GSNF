from model.flow_solvers import GNNFlow
from model.components import Embedding_MLP, Embedding_MLP_GNN, Reconst_Mapper_MLP
from model.gsnf_biclass import GSNF_BiClass
from utils import SolverWrapper_GNN


class ModelFactory:
    def __init__(self, args):
        self.args = args

    def build_flow_solver(self, states_dim):
        hidden_dims = [self.args.hidden_dim] * self.args.hidden_layers
        graph_nodes = int(getattr(self.args, "graph_num_nodes", 0) or states_dim)
        return SolverWrapper_GNN(GNNFlow(
            1,
            graph_nodes,
            self.args.flow_layers,
            hidden_dims,
            self.args.time_net,
            self.args.time_hidden_dim,
            gnn_conv=getattr(self.args, "gnn_flow_conv", "basic"),
            gcn_hidden_dim=getattr(self.args, "gcn_hidden_dim", 32),
            graph_direction=getattr(self.args, "graph_flow_direction", "col"),
        ))

    def init_components(self):
        embedding_nn = Embedding_MLP(
            self.args.variable_num,
            self.args.latent_dim,
            use_delta=getattr(self.args, "use_delta_input", False),
        )
        embedding_nn_gnn = Embedding_MLP_GNN(
            1,
            1,
            use_delta=getattr(self.args, "use_delta_input", False),
        )
        flow_solver = self.build_flow_solver(self.args.latent_dim)
        reconst_mapper = Reconst_Mapper_MLP(
            self.args.latent_dim,
            self.args.variable_num,
        )
        return embedding_nn, embedding_nn_gnn, flow_solver, reconst_mapper

    def initialize_biclass_model(self):
        embedding_nn, embedding_nn_gnn, flow_solver, reconst_mapper = self.init_components()
        return GSNF_BiClass(
            args=self.args,
            embedding_nn=embedding_nn,
            embedding_nn_gnn=embedding_nn_gnn,
            reconst_mapper=reconst_mapper,
            flow_solver=flow_solver,
        )
