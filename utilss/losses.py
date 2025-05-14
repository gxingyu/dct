import torch
from utilss.sde import VPSDE, VESDE, subVPSDE
from utilss.graph_utils import node_flags, mask_x, mask_adjs, gen_noise


def get_score_fn(guidance, sde, model, train=True, continuous=True):
    if not train:
        model.eval()
    model_fn = model
    guidance_fn = guidance
    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        def score_fn(x, adj, flags, t):
            if continuous:
                score = model_fn(x, adj, flags)
                if not train:
                    score += (guidance(x, t)+guidance(adj,t))/2
                std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
            else:
                raise NotImplementedError(f"Discrete not supported")
            score = -score / std[:, None, None]
            return score

    elif isinstance(sde, VESDE):
        def score_fn(x, adj, flags, t):
            if continuous:
                score = model_fn(x, adj, flags)
                if not train:
                    score += (guidance(x, t)+guidance(adj,t))/2
            else:
                raise NotImplementedError(f"Discrete not supported")
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

    return score_fn


def get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=False, continuous=True,
                    likelihood_weighting=False, eps=1e-5):
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(density, guidance_x, guidance_adj, optim_x, optim_adj, model_x, model_adj, x, adj):
        score_fn_x = get_score_fn(None, sde_x, model_x, train=train, continuous=continuous)
        score_fn_adj = get_score_fn(None, sde_adj, model_adj, train=train, continuous=continuous)

        t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps
        flags = node_flags(adj)
        optim_x.zero_grad()
        optim_adj.zero_grad()

        z_x = gen_noise(x, flags, sym=False)
        mean_x, std_x = sde_x.marginal_prob(x, t)
        perturbed_x = mean_x + std_x[:, None, None] * z_x
        out_x = guidance_x(perturbed_x, t)
        loss_x = (torch.mean(out_x) - density)**2
        print(f"loss_x: {loss_x}")
        loss_x.backward(retain_graph=True)
        optim_x.step()

        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        out_adj = guidance_adj(perturbed_adj, t)
        loss_adj = (torch.mean(out_adj) - density)**2
        print(f"loss_adj: {loss_adj}")
        loss_adj.backward(retain_graph=True)
        optim_adj.step()


        score_x = score_fn_x(perturbed_x, perturbed_adj, flags, t)
        score_adj = score_fn_adj(perturbed_x, perturbed_adj, flags, t)
        if not likelihood_weighting:
            losses_x = torch.square(score_x * std_x[:, None, None] + z_x)
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)
            losses_adj = torch.square(score_adj * std_adj[:, None, None] + z_adj)
            losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1)

        else:
            g2_x = sde_x.sde(torch.zeros_like(x), t)[1] ** 2
            losses_x = torch.square(score_x + z_x / std_x[:, None, None])
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1) * g2_x

            g2_adj = sde_adj.sde(torch.zeros_like(adj), t)[1] ** 2
            losses_adj = torch.square(score_adj + z_adj / std_adj[:, None, None])
            losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1) * g2_adj

        return torch.mean(losses_x), torch.mean(losses_adj)

    return loss_fn
