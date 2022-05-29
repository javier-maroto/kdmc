import torch
import numpy as np

from collections.abc import Iterable
from copy import deepcopy


class Attacker:
    """
    Abstract class for adversarial attacks
    """

    def __init__(self, model, device):
        self.model = model
        self.model = self.model.to(device)
        self.device = device

    def attack(self, inputs, targets):
        raise NotImplementedError

    @staticmethod
    def _l2_clip(x, epsilon):
        norm = x.norm(
            p=2, dim=np.arange(start=1, stop=len(x.shape), dtype=int).tolist(), keepdim=True
        )
        norm = torch.max(torch.full_like(norm, 1e-8), norm)
        return x * torch.min(torch.ones_like(norm), epsilon / norm)

    @staticmethod
    def _linf_clip(x, x_min, x_max):
        return torch.max(torch.min(x, x_max), x_min)


class DeepFool(Attacker):
    def __init__(
        self,
        model,
        num_classes=10,
        max_iter=20,
        overshoot=0.02,
        refinement_steps=5,
        Sp=None,
        device=None,
    ):
        """
        Initiates an L2 DeepFool attacker object
        :param model: the model to attack
        :param num_classes: number of classes to consider when checking the boundaries
        :param max_iter: maximum iterations of DeepFool
        :param overshoot: slight increment to cross the boundary
        :param refinement_steps: number of iterations for refining the computed adversarial perturbation
        :param Sp: subspace to look for adversarial examples
        :param device: cpu or gpu operations
        """
        super(DeepFool, self).__init__(model, device)
        self.max_iter = max_iter
        self.overshoot = overshoot
        self.num_classes = num_classes
        self.refinement_steps = refinement_steps
        self.Sp = Sp

    def __call__(self, x):
        return self.batched_attack(x)

    def attack(self, inputs):
        deltas = torch.zeros_like(inputs)

        for i, x in enumerate(inputs):
            if len(x.shape) == 1:
                x = x.to(self.device)[None, :]
            elif len(x.shape) == 2:
                x = x.to(self.device)[None, :, :]
            elif len(x.shape) == 3:
                x = x.to(self.device)[None, :, :, :]

            deltas[i], _, _, _, _ = self._deepfool(x)

            if self.refinement_steps > 0:
                deltas[i] = self._boundary_backtracking(x, deltas[i])

        return deltas.detach()

    def batched_attack(self, inputs):
        deltas = self._batched_deepfool(inputs)[0]
        deltas = self.refine(inputs, deltas)
        return deltas

    def refine(self, inputs, deltas):
        if self.refinement_steps > 0:
            deltas = self._batched_boundary_backtracking(inputs, deltas)
        return deltas

    def _boundary_backtracking(self, point, direction):
        max_radius = direction.norm()
        direction.data = direction / direction.norm()

        output_point = self.model(point)

        # Test boundary
        left = 0
        right = max_radius
        i = 0

        radius = max_radius
        while left < right and i < self.refinement_steps:
            if i == 0:
                radius = right
            else:
                radius = (right + left) / 2
            output = self.model(point + direction * radius)
            if output.argmax() == output_point.argmax():
                left = radius
            else:
                right = radius

            i += 1

        return (direction * radius).detach()

    def _batched_boundary_backtracking(self, points, directions):
        B = points.shape[0]
        
        max_radius = directions.view(B, -1).norm(dim=-1)
        directions.data = directions / max_radius.view(B, 1, 1)

        output_points = self.model(points)

        # Test boundary
        left = torch.zeros_like(max_radius)
        right = max_radius
        i = 0

        radius = max_radius
        finished = True
        while i < self.refinement_steps:
            for j in range(B):
                if left[j] >= right[j]:
                    continue
                if i == 0:
                    radius[j] = right[j]
                else:
                    radius[j] = (right[j] + left[j]) / 2
                    finished = False
            if finished:
                break
            else:
                finished = True
            outputs = self.model(points + directions * radius.view(B, 1, 1))
            for j in range(B):
                if left[j] >= right[j]:
                    continue
                if outputs[j].argmax() == output_points[j].argmax():
                    left[j] = radius[j]
                else:
                    right[j] = radius[j]
            i += 1

        return directions * radius.view(B, 1, 1)

    def _deepfool(self, x):
        x_init = x.clone()
        input_shape = x_init.size()

        f_image = self.model(torch.autograd.Variable(x_init, requires_grad=True)).view((-1,))
        image = f_image.argsort(descending=True)
        image = image[0 : self.num_classes]
        label_orig = image[0]

        pert_image = x_init.clone()

        r = torch.zeros(input_shape).to(self.device)

        label_pert = label_orig
        itr = 0
        while label_pert == label_orig and itr < self.max_iter:

            with torch.cuda.amp.autocast():
                x_curr = torch.autograd.Variable(pert_image, requires_grad=True)
                fs = self.model(x_curr)

            pert = torch.Tensor([np.inf])[0].to(self.device)
            w = torch.zeros(input_shape).to(self.device)

            fs[0, image[0]].backward(retain_graph=True)
            grad_orig = deepcopy(x_curr.grad.data)

            for k in range(1, self.num_classes):
                zero_gradients(x_curr)

                fs[0, image[k]].backward(retain_graph=True)
                cur_grad = deepcopy(x_curr.grad.data)

                w_k = cur_grad - grad_orig
                f_k = (fs[0, image[k]] - fs[0, image[0]]).data

                if self.Sp is None:
                    pert_k = torch.abs(f_k) / w_k.norm()
                else:
                    pert_k = torch.abs(f_k) / torch.matmul(self.Sp.t(), w_k.view([-1, 1])).norm()

                if pert_k < pert:
                    pert = pert_k + 0.0
                    w = w_k + 0.0

            if self.Sp is not None:
                w = torch.matmul(self.Sp, torch.matmul(self.Sp.t(), w.view([-1, 1]))).reshape(
                    w.shape
                )

            r_i = torch.clamp(pert, min=1e-4) * w / w.norm()
            r = r + r_i

            pert_image = pert_image + r_i

            label_pert = torch.argmax(
                self.model(
                    torch.autograd.Variable(x_init + (1 + self.overshoot) * r, requires_grad=False)
                ).data
            ).item()

            itr += 1

        r = r.detach()
        x_init = x_init.detach()

        return (
            (1 + self.overshoot) * r,
            itr,
            label_orig,
            label_pert,
            x_init + (1 + self.overshoot) * r,
        )

    def _batched_deepfool(self, x):
        x_adv = deepcopy(x)
        B = x.shape[0]
        device = x_adv.device
        x_adv.requires_grad = True
        fs = self.model(x_adv)
        iteration = 0

        I = torch.argsort(fs, dim=1, descending=True)[:, :self.num_classes]
        original = I[:, 0]
        current = original.clone()

        cls_idx = lambda k, not_finished: (torch.arange(I.shape[0])[not_finished], I[not_finished, k])

        r_tot = torch.zeros(x.shape, device=device)
        last_iters = torch.full([B], 0, device=device)

        not_finished = (current == original)
        while torch.any(not_finished) and iteration < self.max_iter:
            grads = []
            for k in range(1, self.num_classes):
                grad = torch.autograd.grad(fs[cls_idx(k, not_finished)].mean(), x_adv, retain_graph=True)[0]
                grads.append(grad)
            grads = [torch.autograd.grad(fs[cls_idx(0, not_finished)].mean(), x_adv)[0]] + grads  # Final append where retain_graph=False
            grads = torch.stack(grads)

            pert = torch.full([B], torch.inf, device=device)
            w = torch.zeros_like(x)
            for k in range(1, self.num_classes):
                w_k = grads[k] - grads[0]
                f_k = fs[:, k] - fs[:, 0]
                pert_k = torch.abs(f_k)/w_k.view(B, -1).norm(dim=-1)
                is_better = (pert_k < pert)
                pert[is_better] = pert_k[is_better]
                w[is_better] = w_k[is_better]
            r_tot += pert.view(B, 1, 1) * w / w.view(B, -1).norm(dim=-1).view(B, 1, 1)

            x_adv.data = x + r_tot
            fs = self.model(x_adv)
            current = torch.argmax(fs, dim=-1)
            not_finished = (current == original)
            iteration += 1
            last_iters[not_finished] = iteration
    
        r_tot = (1 + self.overshoot)*r_tot
        return r_tot, last_iters


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, Iterable):
        for elem in x:
            zero_gradients(elem)