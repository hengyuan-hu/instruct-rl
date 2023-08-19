import torch
import torch.nn as nn
from net import PublicLSTMPolicyNet


class PPOAgent(torch.jit.ScriptModule):
    def __init__(
        self,
        ppo_clip,
        ent_weight,
        multi_step,
        gamma,
        device,
        net_type,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        off_belief,
    ):
        super().__init__()
        if net_type == "publ-lstm":
            self.policy_net = PublicLSTMPolicyNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
        else:
            assert False, net_type

        self.net_type = net_type
        self.ppo_clip = ppo_clip
        self.ent_weight = ent_weight
        self.multi_step = multi_step
        self.gamma = gamma
        self.num_lstm_layer = num_lstm_layer
        self.off_belief = off_belief
        if self.off_belief:
            assert self.multi_step == 1

        self.h0_keys: list[str] = list(self.policy_net.get_h0(1).keys())

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> dict[str, torch.Tensor]:
        return self.policy_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        assert overwrite is None
        cloned = type(self)(
            self.ppo_clip,
            self.ent_weight,
            self.multi_step,
            self.gamma,
            device,
            self.net_type,
            self.policy_net.in_dim,
            self.policy_net.hid_dim,
            self.policy_net.out_dim,
            self.num_lstm_layer,
            self.off_belief,
        )
        cloned.load_state_dict(self.state_dict())
        cloned.train(self.training)
        return cloned.to(device)

    @torch.jit.script_method
    def act(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        priv_s = obs["priv_s"]
        legal_move = obs["legal_move"]

        # converge it hid to from batch first to batch second
        # hid size: [batch, num_layer, num_player, dim] -> [num_layer, batch x num_player, dim]
        batch, num_layer, num_player, rnn_dim = obs["h0"].size()
        hid = {}
        for k in self.h0_keys:
            hid[k] = obs[k].transpose(0, 1).flatten(1, 2).contiguous()

        action, log_pa, new_hid = self.policy_net.act(priv_s, legal_move, hid)

        reply = {"a": action.detach().cpu(), "log_pa": log_pa.detach().cpu()}

        # convert hid back to the batch first shape
        # hid size: [num_layer, batch x num_player, dim] -> [batch, num_layer, num_player, dim]
        for k, v in new_hid.items():
            v = v.transpose(0, 1).view(batch, num_layer, num_player, rnn_dim)
            reply[k] = v.detach().cpu()

        return reply

    # @torch.jit.script_method
    # def compute_target(
    #     self, input_: Dict[str, torch.Tensor]
    # ) -> Dict[str, torch.Tensor]:
    #     assert self.multi_step == 1
    #     priv_s = input_["priv_s"]

    #     # hid size: [batch, num_layer, num_player, dim] -> [num_layer, batch x num_player, dim]
    #     hid = {
    #         "h0": input_["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
    #         "c0": input_["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
    #     }
    #     reward = input_["reward"]
    #     terminal = input_["terminal"]

    #     value = self.policy_net.compute_value(priv_s, hid)
    #     target = reward + (1 - terminal) * self.gamma * value
    #     return {"target": target.detach().cpu()}

    def td_error(
        self,
        obs: dict[str, torch.Tensor],
        hid: dict[str, torch.Tensor],
        reply: dict[str, torch.Tensor],
        reward: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_len: torch.Tensor,
        stat,
    ):
        max_seq_len = obs["priv_s"].size(0)
        priv_s = obs["priv_s"]
        legal_move = obs["legal_move"]
        old_log_pa = reply["log_pa"]
        action = reply["a"]  # ah, silly redefiniation of action

        for k, v in hid.items():
            hid[k] = v.flatten(1, 2).contiguous()

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        log_pa, value, ent, p = self.policy_net(priv_s, legal_move, action, hid)

        # mask out dummuy steps at the end of each trajectory
        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()

        if self.off_belief:
            target_value = obs["target"]
        elif self.multi_step >= 80:
            assert bootstrap.sum() == 0
            # note: reward here is the n-step discounted total reward
            target_value = reward
        else:
            target_value = value.clone().detach()
            target_value = torch.cat(
                [target_value[self.multi_step :], target_value[: self.multi_step]], 0
            )
            target_value = (
                reward + bootstrap * self.gamma**self.multi_step * target_value
            )
            target_value = target_value.detach()

        # value_loss: [seq, batch]
        value_loss = nn.functional.smooth_l1_loss(value, target_value, reduction="none")

        adv = (target_value - value).detach()
        # policy loss
        reweighted_pa = torch.exp(log_pa - old_log_pa)
        surr1 = reweighted_pa * adv
        surr2 = (
            torch.clamp(reweighted_pa, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv
        )
        policy_loss = -torch.min(surr1, surr2)

        assert policy_loss.size() == value_loss.size()
        assert policy_loss.size() == ent.size()

        loss = value_loss + policy_loss
        if "llm_prior" not in obs:
            loss  = loss - self.ent_weight * ent
        else:
            pikl_lambda = obs["pikl_lambda"]
            llm_prior_logit = obs["llm_prior"]
            llm_prior_legal_logit = llm_prior_logit - (1 - legal_move) * 1e10
            # llm_prior_legal_logit: [seq_len, batchsize, num_action]
            log_llm_prior = nn.functional.log_softmax(llm_prior_legal_logit, dim=-1)
            log_llm_prior = log_llm_prior * legal_move
            kl_loss = -(log_llm_prior * p).sum(-1)
            # kl_loss: [seq_len, batchsize]
            loss = loss + pikl_lambda * (kl_loss - ent)
            stat["kl_loss/t"].feed(((kl_loss * mask).sum(0) / seq_len).mean().item())

        loss = (loss * mask).sum(0)

        # for logging
        value_loss_t = ((value_loss * mask).sum(0) / seq_len).mean().item()
        policy_loss_t = ((policy_loss * mask).sum(0) / seq_len).mean().item()
        ent_t = ((ent * mask).sum(0) / seq_len).mean().item()
        loss_t = (loss / seq_len).mean().item()
        stat["policy_loss/t"].feed(policy_loss_t)
        stat["value_loss/t"].feed(value_loss_t)
        stat["ent/t"].feed(ent_t)
        stat["loss/t"].feed(loss_t)

        return loss

    def loss(self, batch, stat):
        loss = self.td_error(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.bootstrap,
            batch.seq_len,
            stat,
        )
        stat["game_len"].feed(batch.seq_len.mean())
        return loss
