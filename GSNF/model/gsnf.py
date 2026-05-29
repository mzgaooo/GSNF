import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

import utils
from experiments.utils_metrics import compute_log_normal_pdf
from model.components import Z_to_mu_ReLU, Z_to_std_ReLU


class SegmentGraphPosterior(nn.Module):
    """Segment-level graph posterior used by GSNF.

    Segment observations and missingness summaries parameterize graph
    posteriors. A sampled graph is used per encoder segment, while a pooled
    graph is used by the decoder trajectory.
    """

    def __init__(self, nodes, num_segments=4, hidden_dim=16, pool="learned", eps=1e-8):
        super().__init__()
        self.nodes = int(nodes)
        self.num_segments = max(1, int(num_segments))
        self.pool = pool
        self.eps = eps
        self.node_feature_dim = 4
        self.q_proj = nn.Linear(self.node_feature_dim, hidden_dim)
        self.k_proj = nn.Linear(self.node_feature_dim, hidden_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(self.nodes, hidden_dim),
            nn.ReLU(),
        )
        self.mu_net = nn.Linear(hidden_dim, self.nodes)
        self.std_net = nn.Linear(hidden_dim, self.nodes)
        self.pool_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _segment_bounds(self, length, device):
        boundaries = torch.linspace(
            0, length, self.num_segments + 1, device=device)
        boundaries = boundaries.round().long().clamp(0, length)
        bounds = []
        for idx in range(self.num_segments):
            start = int(boundaries[idx].item())
            end = int(boundaries[idx + 1].item())
            if end <= start:
                end = min(length, start + 1)
            bounds.append((start, end))
        return bounds

    def _node_features(self, x_seg, m_seg, times_seg):
        denom = m_seg.sum(dim=1).clamp_min(self.eps)
        x_mean = (x_seg * m_seg).sum(dim=1) / denom
        obs_rate = m_seg.mean(dim=1)
        rev_mask = torch.flip(m_seg, dims=[1])
        rev_last = rev_mask.argmax(dim=1)
        last_idx = (x_seg.size(1) - 1 - rev_last).unsqueeze(1)
        x_last = torch.gather(x_seg, 1, last_idx).squeeze(1)
        x_last = x_last * m_seg.sum(dim=1).gt(0).to(x_last)

        if times_seg is None:
            time_center = x_mean.new_zeros(x_mean.shape)
        else:
            time_values = times_seg.unsqueeze(-1).expand_as(m_seg)
            observed_time = (time_values * m_seg).sum(dim=1) / denom
            time_center = observed_time

        return torch.stack([x_mean, obs_rate, x_last, time_center], dim=-1)

    def _pool_weights(self, segment_kl, segment_repr):
        if self.pool == "kl_weighted":
            weights = segment_kl / segment_kl.sum(dim=1, keepdim=True).clamp_min(self.eps)
        elif self.pool == "uniform":
            weights = segment_kl.new_full(
                segment_kl.shape, 1.0 / float(segment_kl.size(1)))
        elif self.pool == "learned":
            weights = torch.softmax(self.pool_net(segment_repr).squeeze(-1), dim=1)
        else:
            raise ValueError(f"Unknown graph_pool: {self.pool}")
        return weights

    def _sample_global_graph(self, segment_mu, segment_std, weights):
        weights_expanded = weights[:, :, None, None]
        global_mu = torch.sum(segment_mu * weights_expanded, dim=1)
        second_moment = torch.sum(
            (segment_std.pow(2) + segment_mu.pow(2)) * weights_expanded,
            dim=1)
        global_var = (second_moment - global_mu.pow(2)).clamp_min(self.eps)
        global_std = torch.sqrt(global_var)
        if self.training:
            global_logits = global_mu + torch.randn_like(global_std) * global_std
        else:
            global_logits = global_mu
        global_adj = torch.softmax(global_logits, dim=-1)
        global_kl = 0.5 * (
            global_mu.pow(2) + global_std.pow(2) - 2.0 * global_std.log() - 1.0)
        return global_adj, global_kl.mean(dim=(1, 2))

    def forward(self, x, mask, times=None):
        # x/mask: [B, T, N]
        if x.size(-1) != self.nodes:
            raise ValueError(
                f"Graph posterior expected {self.nodes} nodes, got {x.size(-1)}")

        batch_size, length, _ = x.shape
        bounds = self._segment_bounds(length, x.device)
        segment_adj = []
        segment_kl = []
        segment_repr = []
        segment_mu = []
        segment_std = []
        time_adj = x.new_zeros(batch_size, length, self.nodes, self.nodes)

        for start, end in bounds:
            x_seg = x[:, start:end]
            m_seg = mask[:, start:end]
            times_seg = times[:, start:end] if times is not None else None
            node_features = self._node_features(x_seg, m_seg, times_seg)

            q = self.q_proj(node_features)
            k = self.k_proj(node_features)
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
            attn = torch.softmax(logits, dim=-1)

            h = self.graph_encoder(attn) + self.node_encoder(node_features)
            mu = self.mu_net(h)
            std = F.softplus(self.std_net(h)) + self.eps
            if self.training:
                adj_logits = mu + torch.randn_like(std) * std
            else:
                adj_logits = mu
            adj = torch.softmax(adj_logits, dim=-1)

            kl = 0.5 * (mu.pow(2) + std.pow(2) - 2.0 * std.log() - 1.0)
            kl = kl.mean(dim=(1, 2))

            segment_adj.append(adj)
            segment_kl.append(kl)
            segment_repr.append(h.mean(dim=1))
            segment_mu.append(mu)
            segment_std.append(std)
            time_adj[:, start:end] = adj.unsqueeze(1).expand(
                -1, end - start, -1, -1)

        segment_adj = torch.stack(segment_adj, dim=1)
        segment_kl = torch.stack(segment_kl, dim=1)
        segment_repr = torch.stack(segment_repr, dim=1)
        segment_mu = torch.stack(segment_mu, dim=1)
        segment_std = torch.stack(segment_std, dim=1)
        weights = self._pool_weights(segment_kl, segment_repr)
        global_adj, global_kl = self._sample_global_graph(
            segment_mu, segment_std, weights)

        graph_info = {
            "segment_adj": segment_adj,
            "segment_kl": segment_kl,
            "segment_weights": weights,
            "global_kl": global_kl.mean(),
            "graph_kl": (segment_kl.sum(dim=1) + global_kl).mean(),
        }
        return time_adj, global_adj, graph_info


class GSNF(nn.Module):
    def __init__(
            self,
            args,
            embedding_nn,
            embedding_nn_gnn,
            reconst_mapper,
            flow_solver):

        super().__init__()

        self.args = args
        self.time_start = 0
        self.latent_dim = args.latent_dim
        self.graph_num_nodes = int(
            getattr(args, "graph_num_nodes", 0) or self.latent_dim)
        if self.graph_num_nodes != self.latent_dim:
            raise ValueError(
                "GSNF expects graph_num_nodes to match latent_dim. "
                f"Got graph_num_nodes={self.graph_num_nodes}, latent_dim={self.latent_dim}.")

        self.register_buffer('obsrv_std', torch.tensor([args.obsrv_std]))
        self.register_buffer('mu', torch.tensor([args.prior_mu]))
        self.register_buffer('std', torch.tensor([args.prior_std]))

        self.embedding_nn = embedding_nn
        self.embedding_nn_gnn = embedding_nn_gnn
        self.flow_solver = flow_solver
        self.reconst_mapper = reconst_mapper
        self.z2mu_mapper = Z_to_mu_ReLU(self.latent_dim)
        self.z2std_mapper = Z_to_std_ReLU(self.latent_dim)
        self.graph_posterior = SegmentGraphPosterior(
            self.graph_num_nodes,
            num_segments=getattr(args, "graph_segments", 4),
            hidden_dim=getattr(args, "graph_hidden_dim", 16),
            pool=getattr(args, "graph_pool", "learned"))

    def _observed_time_mask(self, mask, times):
        observed = mask.sum(dim=-1).gt(0)
        observed = observed | times.gt(torch.zeros_like(times))
        return observed

    def _graph_inputs(self, data_in, mask_in, data_embeded, times_in):
        if data_in.size(-1) == self.graph_num_nodes:
            return data_in, mask_in
        graph_mask = mask_in.sum(dim=-1, keepdim=True).gt(0).to(data_embeded)
        graph_mask = graph_mask.expand_as(data_embeded)
        return data_embeded, graph_mask

    def _infer_graphs(self, data_in, mask_in, data_embeded, times_in):
        graph_x, graph_mask = self._graph_inputs(
            data_in, mask_in, data_embeded, times_in)
        return self.graph_posterior(graph_x, graph_mask, times=times_in)

    def _apply_recon_mask_drop(self, data, mask):
        rate = float(getattr(self.args, "recon_mask_drop_rate", 0.0))
        if (not self.training) or rate <= 0.0:
            return data, mask
        if rate >= 1.0:
            rate = 0.999

        observed = mask > 0
        keep = (torch.rand_like(mask) >= rate) | (~observed)
        dropped_mask = mask * keep.to(mask)

        empty_seq = dropped_mask.sum(dim=(1, 2), keepdim=True) <= 0
        if empty_seq.any():
            dropped_mask = torch.where(empty_seq, mask, dropped_mask)

        return data * dropped_mask, dropped_mask

    def _encode_initial_components(self, batch, k_iwae):
        times_in = batch['times_in']
        times_initial = torch.zeros_like(times_in)
        data_in = batch['data_in']
        mask_in = batch['mask_in']
        delta_in = batch.get('delta_in')

        utils.check_mask(data_in, mask_in)
        self.time_start = time.time()
        data_for_encoder, mask_for_encoder = self._apply_recon_mask_drop(data_in, mask_in)

        data_embeded = self.embedding_nn(data_for_encoder, mask_for_encoder, delta_in)
        delta_gnn = delta_in.unsqueeze(-1) if delta_in is not None else None
        data_embeded_h = self.embedding_nn_gnn(
            data_for_encoder.unsqueeze(-1), mask_for_encoder.unsqueeze(-1), delta_gnn)
        if data_embeded_h.size(-2) != self.latent_dim:
            data_embeded_h = data_embeded.unsqueeze(-1)
        adj_time, adj_global, graph_info = self._infer_graphs(
            data_for_encoder, mask_for_encoder, data_embeded, times_in)

        latent = self.flow_solver(
            data_embeded.unsqueeze(-2),
            data_embeded_h.unsqueeze(-3),
            times_in.unsqueeze(-1),
            times_initial.unsqueeze(-1),
            adj_time).squeeze(-2)

        t_exist = times_in.gt(torch.zeros_like(times_in))
        lat_exist = t_exist.unsqueeze(-1).repeat(1, 1, self.latent_dim)
        lat_denom = lat_exist.sum(dim=-2, keepdim=True).clamp_min(1.0)
        lat_mu = torch.sum(latent * lat_exist, dim=-2, keepdim=True) / lat_denom
        lat_variance = torch.sum(
            (latent - lat_mu).pow(2) * lat_exist,
            dim=-2,
            keepdim=True) / lat_denom

        z0_mean = self.z2mu_mapper(latent)
        z0_std = self.z2std_mapper(latent) + 1e-8
        z0_mean = torch.nan_to_num(z0_mean * lat_exist)
        z0_std = torch.nan_to_num(z0_std, nan=1e-8, posinf=1.0, neginf=1e-8)

        t_loss_start = time.time()
        fp_distr = Normal(z0_mean, z0_std)
        kldiv_z0_all = kl_divergence(
            fp_distr, torch.distributions.Normal(self.mu, self.std))
        kldiv_z0 = torch.sum(kldiv_z0_all * lat_exist, (1, 2)) / \
            lat_exist.sum((1, 2)).clamp_min(1.0)
        t_loss_end = time.time()
        self.time_start += t_loss_end - t_loss_start

        z0_mean_iwae = z0_mean.repeat(k_iwae, 1, 1, 1)
        z0_std_iwae = z0_std.repeat(k_iwae, 1, 1, 1)
        initial_state = utils.sample_standard_gaussian(
            z0_mean_iwae, z0_std_iwae)

        if self.args.combine_methods == "average":
            initial_state = torch.sum(
                initial_state * lat_exist, dim=-2, keepdim=True) / lat_denom
        elif self.args.combine_methods == "kl_weighted":
            kl_w = kldiv_z0_all / torch.sum(
                kldiv_z0_all * lat_exist, dim=-2, keepdim=True).clamp_min(1e-8)
            kl_w = (kl_w * lat_exist).repeat(k_iwae, 1, 1, 1)
            initial_state = torch.sum(initial_state * kl_w, dim=-2, keepdim=True)
        else:
            raise NotImplementedError

        return {
            "times_in": times_in,
            "times_initial": times_initial,
            "data_in": data_in,
            "mask_in": mask_in,
            "data_embeded": data_embeded,
            "data_embeded_h": data_embeded_h,
            "lat_exist": lat_exist,
            "lat_variance": lat_variance,
            "kldiv_z0": kldiv_z0,
            "initial_state": initial_state,
            "adj_global": adj_global,
            "graph_info": graph_info,
        }

    def _solve_from_initial(self, initial_state, times_start, times_end, adj):
        initial_state_h = initial_state.view(
            initial_state.shape[0],
            initial_state.shape[1],
            initial_state.shape[2],
            self.latent_dim,
            1)
        return self.flow_solver(
            initial_state,
            initial_state_h,
            times_start.unsqueeze(0),
            times_end.unsqueeze(0),
            adj)

    def _last_observed_indices(self, mask, times):
        if mask.dim() == 4:
            mask = mask[0]
        observed = self._observed_time_mask(mask, times)
        lengths = observed.sum(dim=-1).long().clamp_min(1)
        return lengths - 1

    def _masked_mse(self, pred, truth, mask):
        sq_error = (pred - truth).pow(2) * mask
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return (sq_error.sum(dim=(2, 3)) / denom).mean()

    def _itg_margin(self, initial_state, shifted_state, adj):
        mode = getattr(self.args, "itg_margin_mode", "fixed")
        fixed_margin = float(getattr(self.args, "itg_margin", 1e-6))
        if mode != "lb":
            return initial_state.new_tensor(fixed_margin)

        with torch.no_grad():
            perturb = torch.norm(shifted_state - initial_state, dim=-1).mean()
            try:
                sigma_a = torch.linalg.svdvals(adj.detach()).amin()
            except RuntimeError:
                sigma_a = initial_state.new_tensor(0.0)
            sigma_w = initial_state.new_tensor(1.0)
            try:
                solver = self.flow_solver.solver
                first_block = solver.layers[0].blocks[0]
                sigma_w = torch.linalg.svdvals(
                    first_block.lin_r[0].weight.detach()).amin()
            except Exception:
                pass
            lower_bound = torch.clamp((sigma_a * sigma_w - 1.0) * perturb, min=0.0)
            return torch.maximum(lower_bound, initial_state.new_tensor(fixed_margin))

    def _compute_itg_loss(self, sol_z, times_out, adj, initial_state):
        if float(getattr(self.args, "ratio_itg", 0.0)) == 0.0 or sol_z.size(2) < 2:
            return sol_z.new_zeros(())

        seq_len = sol_z.size(2)
        frac = float(getattr(self.args, "itg_reinit_frac", 0.5))
        t_star_idx = int(round((seq_len - 1) * frac))
        t_star_idx = min(max(1, t_star_idx), seq_len - 1)

        shifted_state = sol_z[:, :, t_star_idx:t_star_idx + 1, :]
        times_abs = times_out[:, t_star_idx:]
        times_start = times_out[:, t_star_idx:t_star_idx + 1].repeat(
            1, times_abs.size(1))
        sol_z_shifted = self._solve_from_initial(
            shifted_state, times_start, times_abs, adj)

        ref = sol_z[:, :, t_star_idx:, :]
        sq_dist = (ref - sol_z_shifted).pow(2).sum(dim=-1).sum(dim=-1)
        margin = self._itg_margin(initial_state, shifted_state, adj)
        return (F.relu(margin.pow(2) - sq_dist) / float(ref.size(2))).mean()

    def _compute_rtg_loss(self, sol_z, times_out, data_out, mask_out, adj):
        if float(getattr(self.args, "ratio_rtg", 0.0)) == 0.0:
            return sol_z.new_zeros(())

        final_state = sol_z[:, :, -1:, :]
        final_time = times_out[:, -1:]
        times_start = final_time.repeat(1, times_out.size(1))
        sol_z_reverse = self._solve_from_initial(
            final_state, times_start, times_out, adj)
        pred_x_reverse = self.reconst_mapper(sol_z_reverse)
        return self._masked_mse(pred_x_reverse, data_out, mask_out)

    def forward(self, batch, k_iwae=1):
        results = dict.fromkeys(['likelihood', 'mse', 'forward_time', 'loss'])

        enc = self._encode_initial_components(batch, k_iwae)
        times_out = enc["times_in"]
        data_out = enc["data_in"]
        mask_out = enc["mask_in"]

        sol_z = self._solve_from_initial(
            enc["initial_state"],
            torch.zeros_like(times_out),
            times_out,
            enc["adj_global"])

        data_out = data_out.repeat(k_iwae, 1, 1, 1)
        mask_out = mask_out.repeat(k_iwae, 1, 1, 1)

        pred_x = self.reconst_mapper(sol_z)
        rec_likelihood = compute_log_normal_pdf(
            data_out, mask_out, pred_x, self.args)

        t_loss_start = time.time()
        ll_z = compute_log_normal_pdf(
            enc["data_embeded"], enc["lat_exist"], sol_z, self.args)
        loss_ll_z = -torch.logsumexp(ll_z, 0).mean(dim=0)

        loss_latent = -torch.logsumexp(
            rec_likelihood - self.args.kl_coef * enc["kldiv_z0"], 0)
        loss_latent = torch.mean(loss_latent, dim=0)

        loss_itg = self._compute_itg_loss(
            sol_z, times_out, enc["adj_global"], enc["initial_state"])
        loss_rtg = self._compute_rtg_loss(
            sol_z, times_out, data_out, mask_out, enc["adj_global"])

        loss = loss_latent + self.args.ratio_zz * loss_ll_z
        loss = loss + self.args.ratio_itg * loss_itg + self.args.ratio_rtg * loss_rtg
        loss = loss + self.args.ratio_graph_kl * enc["graph_info"]["graph_kl"]
        t_loss_end = time.time()
        self.time_start += t_loss_end - t_loss_start

        results["loss"] = loss
        results['likelihood'] = torch.mean(rec_likelihood).detach()
        results['kldiv_z0'] = torch.mean(enc["kldiv_z0"]).detach()
        results['loss_ll_z'] = loss_ll_z.detach()
        results['loss_itg'] = loss_itg.detach()
        results['loss_rtg'] = loss_rtg.detach()
        results['graph_kl'] = enc["graph_info"]["graph_kl"].detach()
        results['graph_kl_loss'] = (
            self.args.ratio_graph_kl * enc["graph_info"]["graph_kl"]).detach()
        results["lat_variance"] = torch.mean(enc["lat_variance"]).detach()

        forward_info = {
            'initial_state': enc["initial_state"],
            'sol_z': sol_z,
            'pred_x': pred_x,
            'adj': enc["adj_global"],
        }
        return results, forward_info

    def forward_classification(self, batch, k_iwae=1):
        results = dict.fromkeys(['likelihood', 'mse', 'forward_time', 'loss'])
        enc = self._encode_initial_components(batch, k_iwae)

        sol_z = None
        if self.args.classifier_input == "zn" or getattr(self.args, "classifier_pool", "none") != "none":
            sol_z = self._solve_from_initial(
                enc["initial_state"],
                torch.zeros_like(enc["times_in"]),
                enc["times_in"],
                enc["adj_global"])

        zero = torch.zeros((), device=enc["data_in"].device)
        results['likelihood'] = zero.detach()
        results['kldiv_z0'] = zero.detach()
        results['loss_ll_z'] = zero.detach()
        results['loss_itg'] = zero.detach()
        results['loss_rtg'] = zero.detach()
        results['graph_kl'] = enc["graph_info"]["graph_kl"].detach()
        results['graph_kl_loss'] = (
            self.args.ratio_graph_kl * enc["graph_info"]["graph_kl"]).detach()
        results["lat_variance"] = torch.mean(enc["lat_variance"]).detach()
        results["loss"] = self.args.ratio_graph_kl * enc["graph_info"]["graph_kl"]

        forward_info = {
            'initial_state': enc["initial_state"],
            'sol_z': sol_z,
            'pred_x': None,
            'adj': enc["adj_global"],
        }
        return results, forward_info

    def run_validation(self, batch):
        return self.forward(batch, k_iwae=self.args.k_iwae)

