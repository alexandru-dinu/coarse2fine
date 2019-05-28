import torch.optim as optim
from torch.nn.utils import clip_grad_norm


class Optim(object):

    def __init__(self, method, lr, alpha, max_grad_norm,
                 lr_decay=1, start_decay_at=None,
                 beta1=0.9, beta2=0.98,
                 opt=None):

        self.last_metric = None
        self.lr = lr
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.opt = opt

        self.params = []
        self.optimizer = None

    def set_parameters(self, params):
        self.params = [p for p in params if p.requires_grad]

        self.optimizer = {
            'sgd'    : optim.SGD(self.params, lr=self.lr),
            'adam'   : optim.Adam(self.params, lr=self.lr, betas=self.betas, eps=1e-9),
            'rmsprop': optim.RMSprop(self.params, lr=self.lr, alpha=self.alpha),
        }.get(self.method, None)

        if self.optimizer is None:
            raise RuntimeError("Invalid optim method: " + self.method)

    def set_learning_rate(self, lr):
        self.lr = lr
        self.optimizer.param_groups[0]['lr'] = self.lr

    def step(self):
        """Compute gradients norm."""
        self._step += 1

        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)

        self.optimizer.step()

    def update_learning_rate(self, metric, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """

        if (self.start_decay_at is not None) and (epoch >= self.start_decay_at):
            self.start_decay = True
        if (self.last_metric is not None) and (metric is not None) and (metric > self.last_metric):
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_metric = metric
        self.optimizer.param_groups[0]['lr'] = self.lr
