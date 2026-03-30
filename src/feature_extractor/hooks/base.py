import contextlib
from dataclasses import dataclass, fields, make_dataclass

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from feature_extractor.logger import init_logging

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
                return int(v.shape[0])
            elif isinstance(v, AbstractBatchResult):
                return v.get_batch_size()
            else:
                continue
        raise ValueError("No tensor found in the result to determine batch size.")


class Hook:
    """Base class for hooks."""

    hook: RemovableHandle
    result: AbstractBatchResult | None
    to_cpu: bool
    positional_args_keys: list[str] | None
    output_keys: list[str] | None
    with_kwargs: bool

    def __init__(
        self,
        module: nn.Module,
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
            assert len(args) <= len(self.positional_args_keys), (
                f"Positional args length {len(args)} exceeds expected "
                f"length {len(self.positional_args_keys)}."
            )
            for k, v in zip(self.positional_args_keys[: len(args)], args):
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
        self.save_result(hook_result)

    def save_result(self, hook_result: dict):
        raise NotImplementedError(
            "save_result method must be implemented by subclasses"
        )

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
