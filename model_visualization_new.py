"""
Visualization tools for MultiTaskNvidiaModel.

This script mirrors the style of model_visualization.py and focuses on
visualizing the architecture defined in model_multitask.py.
"""

import argparse
import os
from pathlib import Path
import shutil
import warnings

import matplotlib.pyplot as plt
import torch

from model_multitask import MultiTaskNvidiaModel

try:
    from torchviz import make_dot
except Exception:  # pragma: no cover - optional dependency
    make_dot = None

try:
    import torchinfo
except Exception:  # pragma: no cover - optional dependency
    torchinfo = None

warnings.filterwarnings("ignore")


class MultiTaskModelVisualizer:
    """Visualization helper for MultiTaskNvidiaModel."""

    def __init__(
        self,
        model,
        input_shape=(3, 224, 224),
        use_speed_input=True,
        num_tl_classes=4,
        device="cpu",
    ):
        self.model = model.to(device)
        self.input_shape = input_shape
        self.use_speed_input = bool(use_speed_input)
        self.num_tl_classes = int(num_tl_classes)
        self.device = device
        self.model.eval()

    def _dummy_inputs(self, batch_size=1):
        images = torch.randn(batch_size, *self.input_shape, device=self.device)
        speed = torch.rand(batch_size, device=self.device) * 40.0
        return images, speed

    def create_computational_graph(self, save_path=None, fmt="png"):
        """Create a computational graph with torchviz if available."""
        if make_dot is None:
            print("torchviz is not installed, skip computational graph.")
            return None

        if shutil.which("dot") is None:
            fallback_dot_dir = Path("D:/Graphviz/bin")
            if fallback_dot_dir.exists():
                current_path = os.environ.get("PATH", "")
                os.environ["PATH"] = f"{fallback_dot_dir};{current_path}"

        images, speed = self._dummy_inputs(batch_size=1)
        images.requires_grad_(True)

        try:
            if self.use_speed_input:
                outputs = self.model(images, prev_speed_kmh=speed)
            else:
                outputs = self.model(images)

            graph_anchor = (
                outputs["steer"].mean()
                + outputs["throttle"].mean()
                + outputs["brake"].mean()
                + outputs["tl_logits"].mean()
                + outputs["stop_logit"].mean()
            )

            if graph_anchor.grad_fn is None:
                raise RuntimeError("Computation graph is empty. Please ensure gradients are enabled.")

            dot = make_dot(graph_anchor, params=dict(self.model.named_parameters()))
            dot.attr(rankdir="TB")
            dot.attr("node", shape="box", style="rounded,filled", fontname="Arial")

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                dot.render(str(save_path), format=fmt, cleanup=True)
                print(f"Computational graph saved to {save_path}.{fmt}")

            return dot
        except Exception as exc:
            print(f"Failed to generate computational graph: {exc}")
            print(f"dot executable detected: {shutil.which('dot')}")
            return None

    def create_model_summary(self, save_path=None):
        """Create text summary for model parameters and key modules."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        lines = [
            "=== MultiTaskNvidiaModel Summary ===",
            f"Input shape: {self.input_shape}",
            f"Use speed input: {self.use_speed_input}",
            f"Traffic light classes: {self.num_tl_classes}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Frozen parameters: {total_params - trainable_params:,}",
            "",
            "=== Key modules ===",
            f"conv_layers: {self.model.conv_layers.__class__.__name__}",
            f"shared_head: {self.model.shared_head.__class__.__name__}",
            f"tl_head: {self.model.tl_head.__class__.__name__}",
            f"stop_head: {self.model.stop_head.__class__.__name__}",
            f"steer_head: {self.model.steer_head.__class__.__name__}",
            f"throttle_head: {self.model.throttle_head.__class__.__name__}",
            f"brake_head: {self.model.brake_head.__class__.__name__}",
        ]

        if torchinfo is not None:
            try:
                lines.append("")
                lines.append("=== torchinfo summary ===")
                images, speed = self._dummy_inputs(batch_size=1)
                if self.use_speed_input:
                    info = torchinfo.summary(
                        self.model,
                        input_data=(images, speed),
                        verbose=0,
                        device=self.device,
                    )
                else:
                    info = torchinfo.summary(
                        self.model,
                        input_data=(images,),
                        verbose=0,
                        device=self.device,
                    )
                lines.append(str(info))
            except Exception as exc:
                lines.append(f"torchinfo summary skipped: {exc}")
        else:
            lines.append("")
            lines.append("torchinfo is not installed, skipped detailed layer summary.")

        text = "\n".join(lines)
        print(text)

        if save_path:
            Path(save_path).write_text(text, encoding="utf-8")
            print(f"Model summary saved to {save_path}")

        return text

    def create_custom_architecture_diagram(self, save_path=None):
        """Draw a custom architecture diagram focused on multitask heads."""
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 10)
        ax.axis("off")

        def add_box(x, y, w, h, text, color):
            rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, weight="bold")

        # Left-to-right main trunk
        add_box(0.8, 4.3, 2.0, 1.4, "Input Image\n(3x224x224)", "#F7E8A4")
        add_box(3.4, 4.3, 2.6, 1.4, "Backbone\nResNet18 Conv", "#A8D5E2")
        add_box(6.6, 4.3, 2.4, 1.4, "Shared Head\n512 -> 128", "#B8E0B4")

        # Right branch: auxiliary tasks and probability transforms
        add_box(10.0, 7.2, 3.6, 1.0, "TL Head\n128 -> num_tl_classes", "#F6C4D0")
        add_box(10.0, 6.0, 3.6, 0.9, "TL Probs\nsoftmax(tl_logits)", "#FDE2E8")
        add_box(10.0, 4.6, 3.6, 1.0, "Stop Head\n128 -> 1", "#F6C4D0")
        add_box(10.0, 3.4, 3.6, 0.9, "Stop Prob\nsigmoid(stop_logit)", "#FDE2E8")

        # Middle-bottom: task features and fusion
        if self.use_speed_input:
            add_box(3.8, 2.3, 2.2, 1.1, "Speed Input\n(B,1)", "#DCC7F7")
            add_box(6.6, 2.3, 2.4, 1.1, "Task Feat\n[shared + speed]", "#CCE5FF")
        else:
            add_box(6.6, 2.3, 2.4, 1.1, "Task Feat\n[shared]", "#CCE5FF")

        add_box(10.0, 2.0, 3.6, 1.0, "Extended Feat (ext)\n[task_feat, tl_probs, stop_prob]", "#D9F2FF")

        # Bottom heads
        add_box(6.6, 0.6, 2.4, 1.0, "Steer Head\n(task_feat only)", "#FFD7A8")
        add_box(10.0, 0.6, 1.6, 1.0, "Throttle\nHead", "#FFD7A8")
        add_box(12.0, 0.6, 1.6, 1.0, "Brake\nHead", "#FFD7A8")

        arrows = [
            ((2.8, 5.0), (3.4, 5.0)),   # input -> backbone
            ((6.0, 5.0), (6.6, 5.0)),   # backbone -> shared
            ((9.0, 5.35), (10.0, 7.7)), # shared -> tl_head
            ((9.0, 5.0), (10.0, 5.1)),  # shared -> stop_head
            ((11.8, 7.2), (11.8, 6.9)), # tl_head -> tl_probs
            ((11.8, 4.6), (11.8, 4.3)), # stop_head -> stop_prob
            ((7.8, 4.3), (7.8, 3.4)),   # shared -> task_feat
            ((9.0, 2.85), (10.0, 2.5)), # task_feat -> ext
            ((11.8, 6.0), (11.8, 3.0)), # tl_probs -> ext
            ((12.5, 3.4), (12.5, 3.0)), # stop_prob -> ext
            ((7.8, 2.3), (7.8, 1.6)),   # task_feat -> steer
            ((10.8, 2.0), (10.8, 1.6)), # ext -> throttle
            ((12.8, 2.0), (12.8, 1.6)), # ext -> brake
        ]

        for start, end in arrows:
            ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.5))

        if self.use_speed_input:
            ax.annotate("", xy=(6.6, 2.85), xytext=(6.0, 2.85), arrowprops=dict(arrowstyle="->", linewidth=1.5))

        ax.text(0.8, 8.9, "TL/Stop auxiliary outputs are converted to probabilities.", fontsize=9)
        ax.text(0.8, 8.5, "Only throttle/brake use the concatenated extended feature ext.", fontsize=9)
        ax.text(0.8, 8.1, "Steer branch consumes task_feat directly.", fontsize=9, color="#7A4F00")

        ax.set_title("MultiTaskNvidiaModel Architecture", fontsize=15, weight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Custom architecture diagram saved to {save_path}")

        plt.show()
        return fig

    def create_visualization_package(self, save_dir="multitask_model_figures"):
        """Generate a set of visualization artifacts in one directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("Creating MultiTask model visualization package...")

        self.create_model_summary(save_path=save_dir / "model_summary.txt")
        self.create_custom_architecture_diagram(save_path=save_dir / "architecture_diagram.png")
        self.create_computational_graph(save_path=save_dir / "computational_graph", fmt="png")

        print(f"All files are saved in: {save_dir}")
        return save_dir


def main():
    parser = argparse.ArgumentParser(description="Visualize MultiTaskNvidiaModel architecture")
    parser.add_argument("--save_dir", type=str, default="multitask_model_figures")
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_tl_classes", type=int, default=4)

    parser.add_argument("--use_speed_input", dest="use_speed_input", action="store_true")
    parser.add_argument("--no_speed_input", dest="use_speed_input", action="store_false")
    parser.set_defaults(use_speed_input=True)

    args = parser.parse_args()

    model = MultiTaskNvidiaModel(
        pretrained=False,
        freeze_features=False,
        use_speed_input=args.use_speed_input,
        num_tl_classes=args.num_tl_classes,
    )

    visualizer = MultiTaskModelVisualizer(
        model=model,
        input_shape=(3, args.input_h, args.input_w),
        use_speed_input=args.use_speed_input,
        num_tl_classes=args.num_tl_classes,
        device=args.device,
    )

    visualizer.create_visualization_package(save_dir=args.save_dir)


if __name__ == "__main__":
    main()
