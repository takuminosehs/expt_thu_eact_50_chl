import torch

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer wrapping an arbitrary base optimizer.
    Compatible with torch.amp.GradScaler (Mixed Precision) when updated manually.
    """
    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            base_optimizer: The class of the base optimizer to wrap (e.g., torch.optim.AdamW).
            rho (float): Neighborhood size for perturbation. Default is 0.05.
            **kwargs: Default arguments passed to the base optimizer.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        # ベースとなるオプティマイザをインスタンス化
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # パラメータグループの参照を同期
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Calculates gradient norm, scales it by rho, adds perturbation to the weights,
        and saves original weights. (Ascends to the local loss maximum)
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                # 現在の重み (w) を退避
                self.state[p]["old_p"] = p.data.clone()
                # 摂動を計算して加算 (w + e(w))
                e_w = (p.grad * scale).to(p)
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Restores original weights and performs the optimization step with the base optimizer.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 退避しておいた元の重み (w) に戻す
                p.data = self.state[p]["old_p"]

        # 摂動が加わった位置で計算された勾配を用いて、元のパラメータ w を更新
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError(
            "SAM requires 2-step execution. Use 'first_step()' and 'second_step()' instead."
        )

    def _grad_norm(self):
        """
        Calculates the L2 norm of the gradients across all parameters.
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups