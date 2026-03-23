import contextlib
from dataclasses import dataclass, fields, make_dataclass

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from feature_extractor.models.architecture import BaseModelArchitecture
from utils.logger import init_logging

logger = init_logging(__name__)


@dataclass
class AbstractResult:
    def __repr__(self):
        msg = self.__class__.__name__ + ":\n"
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                msg += f"\t{k}: {v.shape}\n"
            elif isinstance(v, AbstractResult):
                msg += f"\t{k}: {v.__class__.__name__}\n"
            else:
                msg += f"\t{k}: {v}\n"
        return msg

    def __init__(self, **kwargs):
        # Get the field names from the dataclass
        field_names = {f.name for f in fields(self.__class__)}
        for key, value in kwargs.items():
            if key in field_names:
                setattr(self, key, value)
        # Handle ignored/unexpected keys
        ignored_keys = set(kwargs) - field_names
        if ignored_keys:
            logger.warn_once(f"Ignored unexpected keys: {ignored_keys}")

    @classmethod
    def init_all(cls, **kwargs):
        for key, value in kwargs.items():
            setattr(cls, key, value)


@dataclass(repr=False, init=False)
class AbstractBatchResult(AbstractResult):
    def unbatch(self) -> list[AbstractResult]:
        for k, v in self.__dict__.items():
            if isinstance(v, AbstractBatchResult):
                setattr(self, k, v.unbatch())
            elif isinstance(v, (torch.Tensor | list)):
                continue
            else:
                raise ValueError(f"Unexpected type: {type(v)}")

        results = []
        new_class_name = self.__class__.__name__.replace("Batch", "")
        new_class_fields = [(f.name, f.type) for f in fields(self)]
        new_class = make_dataclass(
            new_class_name,
            fields=new_class_fields,
            bases=(AbstractResult,),
            repr=False,
            init=False,
        )

        for i in range(self.get_batch_size()):
            results.append(new_class(**{k: v[i] for k, v in self.__dict__.items()}))

        return results

    def get_batch_size(self) -> int:
        """Get the batch size."""
        for v in self.__dict__.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
            elif isinstance(v, AbstractBatchResult):
                return v.get_batch_size()


class Hook:
    """Base class for hooks."""

    hook: RemovableHandle
    result: AbstractBatchResult
    result_class: type[AbstractBatchResult]
    to_cpu: bool
    positional_args_keys: list[str]
    output_keys: list[str]
    with_kwargs: bool

    def __init__(
        self,
        module: nn.Module,
        result_class: AbstractBatchResult,
        to_cpu: bool = True,
        with_args: None | list[str] = None,
        with_kwargs: bool = False,
        with_output: None | list[str] = None,
    ) -> None:
        # Register a forward hook
        self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)

        # Register keys for positional args and output
        self.positional_args_keys = with_args
        self.output_keys = with_output
        self.with_kwargs = with_kwargs

        self.result = None
        self.result_class = result_class
        self.to_cpu = to_cpu

    def hook_fn(
        self, module: nn.Module, args: tuple, kwargs: dict, output: tuple
    ) -> None:
        """Forward hook function to capture inputs and outputs.

        Parameters
        ----------
        module : nn.Module
            Module with hook
        args : tuple
            Position arguments passed to the module
        kwargs : dict
            Keyword arguments passed to the module
        output : Tensor
            Output of the module
        """

        hook_result = {}

        # Add positional arguments to the hook result
        if self.positional_args_keys:
            assert len(self.positional_args_keys) == len(args), (
                f"Positional args length {len(args)} does not match expected "
                f"length {len(self.positional_args_keys)}."
            )
            for k, v in zip(self.positional_args_keys, args):
                assert k not in kwargs, f"Key {k} already exists in kwargs."
                hook_result[k] = (
                    v.cpu().clone()
                    if self.to_cpu and isinstance(v, torch.Tensor)
                    else v
                )

        # Add keyword arguments to the hook result
        if self.with_kwargs:
            hook_result.update(
                {
                    k: v.cpu().clone()
                    if self.to_cpu and isinstance(v, torch.Tensor)
                    else v
                    for k, v in kwargs.items()
                }
            )

        # Add output to the hook result
        if self.output_keys is not None:
            if isinstance(output, tuple):
                assert len(output) == len(self.output_keys), (
                    f"Output tuple length {len(output)} does not match expected "
                    f"length {len(self.output_keys)}"
                )
                for k, v in zip(self.output_keys, output):
                    assert k not in hook_result, f"Key {k} already exists in kwargs."
                    hook_result[k] = (
                        v.cpu().clone()
                        if self.to_cpu and isinstance(v, torch.Tensor)
                        else v
                    )
                    if k == "attn_weights":
                        print("attn_weights: ", v)
            else:
                assert len(self.output_keys) == 1, (
                    f"Output keys length {len(self.output_keys)} does not match expected "
                    f"length 1 for single output."
                )
                hook_result[self.output_keys[0]] = (
                    output.cpu().clone()
                    if self.to_cpu and isinstance(output, torch.Tensor)
                    else output
                )

        self.result = self.result_class(**hook_result)

    def remove(self):
        """Remove the hook."""
        self.hook.remove()

    @classmethod
    @contextlib.contextmanager
    def context(cls, hooks: "list[Hook]"):
        """Context manager to use the hook."""
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()


class HookManager:
    """Base class for managing multiple hooks for a model component."""

    def __init__(
        self,
        model: nn.Module,
        architecture: BaseModelArchitecture | None = None,
    ) -> None:
        """Initialize the hook manager.

        Parameters
        ----------
        model : nn.Module
            Model instance that owns the hooked modules.
        architecture : BaseModelArchitecture | None
            Optional architecture metadata; when None, defaults are used.
        """
        self._model = model
        self._architecture = architecture or BaseModelArchitecture()
        self._hooks: list[Hook] = []

    def install(self) -> None:
        """Install hooks (no-op by default).

        Subclasses should override this to register their specific hooks.
        """

    def reset(self) -> None:
        """Reset any cached hook state."""
        for hook in self._hooks:
            hook.result = None

    def remove(self) -> None:
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def validate_layer_count(self, actual_layer_count: int) -> None:
        """Validate hook count matches expected layers (no-op by default).

        Parameters
        ----------
        actual_layer_count : int
            Actual number of model layers to validate against.

        Subclasses should override this method when they install per-layer
        hooks and want to assert the hook count matches the model layers.
        """
