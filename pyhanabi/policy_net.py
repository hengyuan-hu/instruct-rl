import torch
import torch.nn as nn
from typing import Tuple, Dict


class PublicLSTMPolicyNet(torch.jit.ScriptModule):
    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        assert len(in_dim) == 3
        self.in_dim = in_dim
        self.priv_in_dim = in_dim[1]
        self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        self.priv_net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )

        ff_layers = [nn.Linear(self.publ_in_dim, self.hid_dim), nn.ReLU()]
        for _ in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.publ_net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)
        self.fc_v = nn.Linear(self.hid_dim, 1)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2
        publ_s = priv_s[:, 125:]
        priv_s = priv_s.unsqueeze(0)
        publ_s = publ_s.unsqueeze(0)

        x = self.publ_net(publ_s)
        publ_o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        a = self.fc_a(o).squeeze(0)

        legal_a = a - (1 - legal_move) * 1e10
        prob_a = nn.functional.softmax(legal_a, 1)
        log_pa = nn.functional.log_softmax(legal_a, 1)
        if self.training:
            action = prob_a.multinomial(1)
        else:
            action = legal_a.argmax(1, keepdim=True)

        log_pa = log_pa.gather(1, action).squeeze(1)
        action = action.squeeze(1)
        assert log_pa.size() == action.size()

        return action, log_pa, {"h0": h, "c0": c}

    @torch.jit.script_method
    def compute_value(
        self,
        priv_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        -> value: [batch]
        """
        assert priv_s.dim() == 2, "dim = 2, [batch, dim]"
        publ_s = priv_s[:, 125:]

        priv_s = priv_s.unsqueeze(0)
        publ_s = publ_s.unsqueeze(0)

        x = self.publ_net(publ_s)
        publ_o, _ = self.lstm(x, (hid["h0"], hid["c0"]))

        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        v = self.fc_v(o).squeeze(2)
        return v.squeeze(0)

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        -> log_pa: [seq_len(optional), batch]
        -> value: [seq_len(optional), batch]
        -> ent: [seq_len(optional), batch]
        """
        assert priv_s.dim() == 3, "dim = 3, [seq_len, batch, dim]"
        publ_s = priv_s[:, :, 125:]

        x = self.publ_net(publ_s)
        if len(hid) == 0:
            publ_o, _ = self.lstm(x)
        else:
            publ_o, _ = self.lstm(x, (hid["h0"], hid["c0"]))

        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        logit_a = self.fc_a(o)
        v = self.fc_v(o).squeeze(2)

        # logit_a: [seq_len, batch, num_action]
        logit_a = logit_a - (1 - legal_move) * 1e10
        p = nn.functional.softmax(logit_a, 2)
        log_p = p.clamp(min=1e-5).log()
        log_pa = log_p.gather(2, action.unsqueeze(2)).squeeze(2)

        ent = (-log_p * p).sum(2)
        return log_pa, v, ent, p
