import torch
from torch import nn
import utils


def pred_loss(logp, gtruth, seq_len):
    """
    logit: [seq_len, batch, hand_size, bits_per_card]
    gtruth: [seq_len, batch, hand_size, bits_per_card]
        one-hot, can be all zero if no card for that position
    """
    assert logp.size() == gtruth.size()
    logp = (logp * gtruth).sum(3)
    hand_size = gtruth.sum(3).sum(2).clamp(min=1e-5)
    logp_per_card = logp.sum(2) / hand_size
    xent = -logp_per_card.sum(0)
    assert seq_len.size() == xent.size()
    avg_xent = xent / seq_len
    nll_per_card = -logp_per_card
    return xent, avg_xent, nll_per_card


class ARBeliefModel(nn.Module):
    def __init__(self, device, in_dim, hid_dim, hand_size, out_dim):
        """
        mode: priv: private belief prediction
              publ: public/common belief prediction
        """
        super().__init__()
        self.device = device
        self.input_key = "priv_s"
        self.ar_input_key = "own_hand_ar_in"
        self.ar_target_key = "own_hand"

        self.in_dim = in_dim
        self.hand_size = hand_size
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layer = 2

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.emb = nn.Linear(25, self.hid_dim // 8, bias=False)
        self.auto_regress = nn.LSTM(
            self.hid_dim + self.hid_dim // 8,
            self.hid_dim,
            num_layers=1,
            batch_first=True,
        ).to(device)
        self.auto_regress.flatten_parameters()

        self.fc = nn.Linear(self.hid_dim, self.out_dim)

    def get_h0(self) -> dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @classmethod
    def load(cls, weight_file, device, hand_size):
        state_dict = torch.load(weight_file)
        state_dict = utils.process_compiled_state_dict(state_dict)
        keys = list(state_dict.keys())
        hid_dim, in_dim = state_dict[keys[0]].size()
        out_dim = state_dict[keys[-1]].size(0)
        model = cls(device, in_dim, hid_dim, hand_size, out_dim)
        model.load_state_dict(state_dict)
        model = model.to(device)
        return model

    def forward(self, x, ar_card_in):
        o, _ = self.lstm(self.net(x))

        seq, bsize, _ = ar_card_in.size()
        ar_card_in = ar_card_in.view(seq * bsize, self.hand_size, 25)

        ar_emb_in = self.emb(ar_card_in)
        # ar_card_in: [seq * batch, 5, 64]
        # o: [seq, batch, 512]
        o = o.view(seq * bsize, self.hid_dim)
        o = o.unsqueeze(1).expand(seq * bsize, self.hand_size, self.hid_dim)
        ar_in = torch.cat([ar_emb_in, o], 2)
        ar_out, _ = self.auto_regress(ar_in)

        logit = self.fc(ar_out)
        logit = logit.view(seq, bsize, self.hand_size, -1)
        return logit

    def loss(self, batch, beta=1):
        logit = self.forward(batch.obs[self.input_key], batch.obs[self.ar_input_key])
        logit = logit * beta
        logp = nn.functional.log_softmax(logit, 3)
        gtruth = batch.obs[self.ar_target_key]
        gtruth = gtruth.view(logp.size())
        seq_len = batch.seq_len
        xent, avg_xent, nll_per_card = pred_loss(logp, gtruth, seq_len)

        # v0: [seq, batch, hand_size, bit_per_card]
        v0 = batch.obs["priv_ar_v0"]
        v0 = v0.view(v0.size(0), v0.size(1), self.hand_size, 35)[:, :, :, :25]
        logv0 = v0.clamp(min=1e-6).log()
        _, avg_xent_v0, _ = pred_loss(logv0, gtruth, seq_len)
        return xent, avg_xent, avg_xent_v0, nll_per_card

    def observe(self, priv_s: torch.Tensor, hid: dict[str, torch.Tensor]):
        x = self.net(priv_s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        return {"h0": h, "c0": c}

    def sample(
        self, priv_s, hid: dict[str, torch.Tensor], num_sample
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.net(priv_s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        # o: [seq_len(1), dim]
        seq, hid_dim = o.size()

        assert seq == 1, "seqlen should be 1"
        # assert bsize == 1, "batchsize for BeliefModel.sample should be 1"
        # o = o.view(1, hid_dim)
        o = o.unsqueeze(1).expand(seq, num_sample, hid_dim)

        in_t = torch.zeros(1, num_sample, hid_dim // 8, device=o.device)
        shape = (1, num_sample, self.hid_dim)
        ar_hid = (
            torch.zeros(*shape, device=o.device),
            torch.zeros(*shape, device=o.device),
        )
        sample_list = []
        for i in range(self.hand_size):
            ar_in = torch.cat([in_t, o], 2).view(num_sample, 1, -1)
            ar_out, ar_hid = self.auto_regress(ar_in, ar_hid)
            logit = self.fc(ar_out.squeeze(1))
            prob = nn.functional.softmax(logit, 1)
            sample_t = prob.multinomial(1)
            sample_t = sample_t.view(num_sample, 1)
            onehot_sample_t = torch.zeros(num_sample, 25, device=sample_t.device)
            onehot_sample_t.scatter_(1, sample_t, 1)
            in_t = self.emb(onehot_sample_t).unsqueeze(0)
            sample_list.append(sample_t)

        sample = torch.stack(sample_list, 2)

        hid = {"h0": h, "c0": c}
        return sample, hid
