from __future__ import annotations
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Tuple, Literal, TypeVar, overload, Any
from uuid import uuid4
import json

from .image import Bounds, Extent, Image, ImageCollection
from .resources import Arch, ControlMode
from .util import base_type_match, client_logger as log


class ComfyRunMode(Enum):
    runtime = 0  # runs as part of same process, transfer images in memory
    server = 1  # runs as a server, transfer images via base64 or websocket


class Output(NamedTuple):
    node: int
    output: int


T = TypeVar("T")
Output2 = Tuple[Output, Output]
Output3 = Tuple[Output, Output, Output]
Output4 = Tuple[Output, Output, Output, Output]
Input = int | float | bool | str | Output


class ComfyNode(NamedTuple):
    id: int
    type: str
    inputs: dict[str, Input]

    @overload
    def input(self, key: str, default: T) -> T: ...

    @overload
    def input(self, key: str, default: None = None) -> Input | None: ...

    def input(self, key: str, default: T | None = None) -> T | Input | None:
        result = self.inputs.get(key, default)
        assert default is None or base_type_match(result, default)
        return result

    def output(self, index=0) -> Output:
        return Output(int(self.id), index)


class ComfyWorkflow:
    """Builder for workflows which can be sent to the ComfyUI prompt API."""

    def __init__(self, node_inputs: dict | None = None, run_mode=ComfyRunMode.server):
        self.root: dict[str, dict] = {}
        self.images: dict[str, Image | ImageCollection] = {}
        self.node_count = 0
        self.sample_count = 0
        self._cache: dict[str, Output | Output2 | Output3 | Output4] = {}
        self._nodes_inputs: dict[str, dict[str, Any]] = node_inputs or {}
        self._run_mode: ComfyRunMode = run_mode

    @staticmethod
    def import_graph(existing: dict, node_inputs: dict):
        w = ComfyWorkflow(node_inputs)
        existing = _convert_ui_workflow(existing, node_inputs)
        node_map: dict[str, str] = {}
        queue = list(existing.keys())
        while queue:
            id = queue.pop(0)
            node = deepcopy(existing[id])
            class_type = node.get("class_type")
            if class_type is None:
                log.warning(f"Workflow import: Node {id} is not installed, aborting.")
                return w
            if node_inputs and class_type not in node_inputs:
                raise ValueError(
                    f"Workflow contains a node of type {class_type} which is not installed on the ComfyUI server."
                )
            edges = [e for e in node["inputs"].values() if isinstance(e, list)]
            if any(e[0] not in node_map for e in edges):
                queue.append(id)  # requeue node if an input is not yet mapped
                continue

            for e in edges:
                e[0] = node_map[e[0]]
            node_map[id] = str(w.node_count)
            w.root[str(w.node_count)] = node
            w.node_count += 1
        return w

    @staticmethod
    def from_dict(existing: dict):
        w = ComfyWorkflow()
        w.root = existing
        w.node_count = len(w.root)
        return w

    def add_default_values(self, node_name: str, args: dict):
        if node_inputs := _inputs_for_node(self._nodes_inputs, node_name, "required"):
            for k, v in node_inputs.items():
                if k not in args:
                    if len(v) == 1 and isinstance(v[0], list) and len(v[0]) > 0:
                        # enum type, use first value in list of possible values
                        args[k] = v[0][0]
                    elif len(v) > 1 and isinstance(v[1], dict):
                        # other type, try to access default value
                        default = v[1].get("default", None)
                        if default is not None:
                            args[k] = default
        return args

    def input_type(self, class_type: str, input_name: str) -> tuple | None:
        if inputs := _inputs_for_node(self._nodes_inputs, class_type):
            return inputs.get(input_name)
        return None

    def dump(self, filepath: str | Path):
        filepath = Path(filepath)
        if filepath.suffix != ".json":
            filepath = filepath / "workflow.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.root, f, indent=4)

    @overload
    def add(self, class_type: str, output_count: Literal[1], **inputs) -> Output: ...

    @overload
    def add(self, class_type: str, output_count: Literal[2], **inputs) -> Output2: ...

    @overload
    def add(self, class_type: str, output_count: Literal[3], **inputs) -> Output3: ...

    @overload
    def add(self, class_type: str, output_count: Literal[4], **inputs) -> Output4: ...

    def add(self, class_type: str, output_count: int, **inputs):
        def normalize(x):
            return [str(x.node), x.output] if isinstance(x, Output) else x

        inputs = self.add_default_values(class_type, inputs)
        self.node_count += 1
        self.root[str(self.node_count)] = {
            "class_type": class_type,
            "inputs": {k: normalize(v) for k, v in inputs.items()},
        }
        output = tuple(Output(self.node_count, i) for i in range(output_count))
        return output[0] if output_count == 1 else output

    @overload
    def add_cached(self, class_type: str, output_count: Literal[1], **inputs) -> Output: ...

    @overload
    def add_cached(self, class_type: str, output_count: Literal[3], **inputs) -> Output3: ...

    def add_cached(self, class_type: str, output_count: Literal[1] | Literal[3], **inputs):
        key = class_type + str(inputs)
        result = self._cache.get(key, None)
        if result is None:
            result = self.add(class_type, output_count, **inputs)
            self._cache[key] = result
        return result

    def remove(self, node_id: int):
        del self.root[str(node_id)]

    def node(self, node_id: int):
        inputs = self.root[str(node_id)]["inputs"]
        inputs = {
            k: Output(int(v[0]), v[1]) if isinstance(v, list) else v for k, v in inputs.items()
        }
        return ComfyNode(node_id, self.root[str(node_id)]["class_type"], inputs)

    def copy(self, node: ComfyNode):
        return self.add(node.type, 1, **node.inputs)

    def find(self, type: str):
        return (self.node(int(k)) for k, v in self.root.items() if v["class_type"] == type)

    def find_connected(self, output: Output):
        for node in self:
            for input_name, input_value in node.inputs.items():
                if input_value == output:
                    yield node, input_name

    def guess_sample_count(self):
        self.sample_count = sum(
            int(value)
            for node in self
            for name, value in node.inputs.items()
            if name == "steps" and isinstance(value, (int, float))
        )
        return self.sample_count

    def __iter__(self):
        return iter(self.node(int(k)) for k in self.root.keys())

    def __contains__(self, node: ComfyNode):
        return any(n == node for n in self)

    def _add_image(self, image: Image | ImageCollection):
        id = str(uuid4())
        self.images[id] = image
        return id

    # Nodes

    def ksampler(
        self,
        model: Output,
        positive: Output,
        negative: Output,
        latent_image: Output,
        sampler="dpmpp_2m_sde_gpu",
        scheduler="normal",
        steps=20,
        cfg=7.0,
        denoise=1.0,
        seed=1234,
    ):
        self.sample_count += steps
        return self.add(
            "KSampler",
            1,
            seed=seed,
            sampler_name=sampler,
            scheduler=scheduler,
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
        )

    def ksampler_advanced(
        self,
        model: Output,
        positive: Output,
        negative: Output,
        latent_image: Output,
        sampler="dpmpp_2m_sde_gpu",
        scheduler="normal",
        steps=20,
        start_at_step=0,
        cfg=7.0,
        seed=-1,
    ):
        self.sample_count += steps - start_at_step

        return self.add(
            "KSamplerAdvanced",
            1,
            noise_seed=seed,
            sampler_name=sampler,
            scheduler=scheduler,
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            steps=steps,
            start_at_step=start_at_step,
            end_at_step=steps,
            cfg=cfg,
            add_noise="enable",
            return_with_leftover_noise="disable",
        )

    def sampler_custom_advanced(
        self,
        model: Output,
        positive: Output,
        negative: Output,
        latent_image: Output,
        model_version: Arch,
        sampler="dpmpp_2m_sde_gpu",
        scheduler="normal",
        steps=20,
        start_at_step=0,
        cfg=7.0,
        seed=-1,
    ):
        self.sample_count += steps - start_at_step

        if model_version is Arch.flux:
            positive = self.flux_guidance(positive, cfg if cfg > 1 else 3.5)
            guider = self.basic_guider(model, positive)
        else:
            guider = self.cfg_guider(model, positive, negative, cfg)

        sigmas = self.scheduler_sigmas(model, scheduler, steps, model_version)
        if start_at_step > 0:
            _, sigmas = self.split_sigmas(sigmas, start_at_step)

        return self.add(
            "SamplerCustomAdvanced",
            output_count=2,
            noise=self.random_noise(seed),
            guider=guider,
            sampler=self.sampler_select(sampler),
            sigmas=sigmas,
            latent_image=latent_image,
        )[1]

    def scheduler_sigmas(
        self, model: Output, scheduler="normal", steps=20, model_version=Arch.sdxl
    ):
        if scheduler in ("align_your_steps", "ays"):
            assert model_version is Arch.sd15 or model_version.is_sdxl_like

            if model_version is Arch.sd15:
                model_type = "SD1"
            else:
                model_type = "SDXL"

            return self.add(
                "AlignYourStepsScheduler",
                output_count=1,
                steps=steps,
                model_type=model_type,
                denoise=1.0,
            )
        elif scheduler == "gits":
            return self.add(
                "GITSScheduler",
                output_count=1,
                steps=steps,
                coeff=1.2,
                denoise=1.0,
            )
        elif scheduler in ("polyexponential", "poly_exponential"):
            return self.add(
                "PolyexponentialScheduler",
                output_count=1,
                steps=steps,
                sigma_max=14.614642,
                sigma_min=0.0291675,
                rho=1.0,
            )
        elif scheduler == "laplace":
            return self.add(
                "LaplaceScheduler",
                output_count=1,
                steps=steps,
                sigma_max=14.614642,
                sigma_min=0.0291675,
                mu=0.0,
                beta=0.5,
            )
        else:
            return self.add(
                "BasicScheduler",
                output_count=1,
                model=model,
                scheduler=scheduler,
                steps=steps,
                denoise=1.0,
            )

    def split_sigmas(self, sigmas: Output, step=0):
        return self.add(
            "SplitSigmas",
            output_count=2,
            sigmas=sigmas,
            step=step,
        )

    def basic_guider(self, model: Output, positive: Output):
        return self.add("BasicGuider", 1, model=model, conditioning=positive)

    def cfg_guider(self, model: Output, positive: Output, negative: Output, cfg=7.0):
        return self.add(
            "CFGGuider",
            output_count=1,
            model=model,
            positive=positive,
            negative=negative,
            cfg=cfg,
        )

    def flux_guidance(self, conditioning: Output, guidance=3.5):
        return self.add("FluxGuidance", 1, conditioning=conditioning, guidance=guidance)

    def random_noise(self, noise_seed=-1):
        return self.add_cached(
            "RandomNoise",
            output_count=1,
            noise_seed=noise_seed,
        )

    def sampler_select(self, sampler_name="dpmpp_2m_sde_gpu"):
        if sampler_name == "euler_cfgpp":
            return self.add_cached(
                "SamplerEulerCFGpp",
                output_count=1,
                version="regular",
            )
        else:
            return self.add_cached(
                "KSamplerSelect",
                output_count=1,
                sampler_name=sampler_name,
            )

    def differential_diffusion(self, model: Output):
        return self.add("DifferentialDiffusion", 1, model=model)

    def model_sampling_discrete(self, model: Output, sampling: str, zsnr=False):
        return self.add("ModelSamplingDiscrete", 1, model=model, sampling=sampling, zsnr=zsnr)

    def model_sampling_sd3(self, model: Output, shift=3.0):
        return self.add("ModelSamplingSD3", 1, model=model, shift=shift)

    def rescale_cfg(self, model: Output, multiplier=0.7):
        return self.add("RescaleCFG", 1, model=model, multiplier=multiplier)

    def load_checkpoint(self, checkpoint: str):
        return self.add_cached("CheckpointLoaderSimple", 3, ckpt_name=checkpoint)

    def load_diffusion_model(self, model_name: str):
        if model_name.endswith(".gguf"):
            return self.add_cached("UnetLoaderGGUF", 1, unet_name=model_name)
        return self.add_cached("UNETLoader", 1, unet_name=model_name, weight_dtype="default")

    def load_clip(self, clip_name: str, type: str):
        return self.add_cached("CLIPLoader", 1, clip_name=clip_name, type=type)

    def load_dual_clip(self, clip_name1: str, clip_name2: str, type: str):
        node = "DualCLIPLoader"
        if any(f.endswith(".gguf") for f in (clip_name1, clip_name2)):
            node = "DualCLIPLoaderGGUF"
        return self.add_cached(node, 1, clip_name1=clip_name1, clip_name2=clip_name2, type=type)

    def load_triple_clip(self, clip_name1: str, clip_name2: str, clip_name3: str):
        node = "TripleCLIPLoader"
        if any(f.endswith(".gguf") for f in (clip_name1, clip_name2, clip_name3)):
            node = "TripleCLIPLoaderGGUF"
        return self.add_cached(
            node, 1, clip_name1=clip_name1, clip_name2=clip_name2, clip_name3=clip_name3
        )

    def load_vae(self, vae_name: str):
        return self.add_cached("VAELoader", 1, vae_name=vae_name)

    def load_controlnet(self, controlnet: str):
        return self.add_cached("ControlNetLoader", 1, control_net_name=controlnet)

    def load_clip_vision(self, clip_name: str):
        return self.add_cached("CLIPVisionLoader", 1, clip_name=clip_name)

    def load_ip_adapter(self, ipadapter_file: str):
        return self.add_cached("IPAdapterModelLoader", 1, ipadapter_file=ipadapter_file)

    def load_upscale_model(self, model_name: str):
        return self.add_cached("UpscaleModelLoader", 1, model_name=model_name)

    def load_style_model(self, model_name: str):
        return self.add_cached("StyleModelLoader", 1, style_model_name=model_name)

    def load_lora_model(self, model: Output, lora_name: str, strength: float):
        return self.add(
            "LoraLoaderModelOnly", 1, model=model, lora_name=lora_name, strength_model=strength
        )

    def load_lora(self, model: Output, clip: Output, lora_name, strength_model, strength_clip):
        return self.add(
            "LoraLoader",
            2,
            model=model,
            clip=clip,
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_clip,
        )

    def load_insight_face(self):
        return self.add_cached(
            "IPAdapterInsightFaceLoader", 1, provider="CPU", model_name="buffalo_l"
        )

    def load_inpaint_model(self, model_name: str):
        return self.add_cached("INPAINT_LoadInpaintModel", 1, model_name=model_name)

    def load_fooocus_inpaint(self, head: str, patch: str):
        return self.add_cached("INPAINT_LoadFooocusInpaint", 1, head=head, patch=patch)

    def empty_latent_image(self, extent: Extent, arch: Arch, batch_size=1):
        w, h = extent.width, extent.height
        if arch in [Arch.sd3, Arch.flux]:
            return self.add("EmptySD3LatentImage", 1, width=w, height=h, batch_size=batch_size)
        return self.add("EmptyLatentImage", 1, width=w, height=h, batch_size=batch_size)

    def clip_set_last_layer(self, clip: Output, clip_layer: int):
        return self.add("CLIPSetLastLayer", 1, clip=clip, stop_at_clip_layer=clip_layer)

    def clip_text_encode(self, clip: Output, text: str | Output):
        return self.add("CLIPTextEncode", 1, clip=clip, text=text)

    def conditioning_area(self, conditioning: Output, area: Bounds, strength=1.0):
        return self.add(
            "ConditioningSetArea",
            1,
            conditioning=conditioning,
            x=area.x,
            y=area.y,
            width=area.width,
            height=area.height,
            strength=strength,
        )

    def conditioning_set_mask(self, conditioning: Output, mask: Output, strength=1.0):
        return self.add(
            "ConditioningSetMask",
            1,
            conditioning=conditioning,
            mask=mask,
            strength=strength,
            set_cond_area="default",
        )

    def conditioning_combine(self, a: Output, b: Output):
        return self.add("ConditioningCombine", 1, conditioning_1=a, conditioning_2=b)

    def conditioning_average(self, a: Output, b: Output, strength_a: float):
        return self.add(
            "ConditioningAverage",
            1,
            conditioning_to=a,
            conditioning_from=b,
            conditioning_to_strength=strength_a,
        )

    def conditioning_step_range(self, conditioning: Output, range: tuple[float, float]):
        return self.add(
            "ConditioningSetTimestepRange",
            1,
            conditioning=conditioning,
            start=range[0],
            end=range[1],
        )

    def conditioning_zero_out(self, conditioning: Output):
        return self.add("ConditioningZeroOut", 1, conditioning=conditioning)

    def instruct_pix_to_pix_conditioning(
        self, positive: Output, negative: Output, vae: Output, pixels: Output
    ):
        return self.add(
            "InstructPixToPixConditioning",
            3,
            positive=positive,
            negative=negative,
            vae=vae,
            pixels=pixels,
        )

    def background_region(self, conditioning: Output):
        return self.add("ETN_BackgroundRegion", 1, conditioning=conditioning)

    def define_region(self, regions: Output, mask: Output, conditioning: Output):
        return self.add(
            "ETN_DefineRegion", 1, regions=regions, mask=mask, conditioning=conditioning
        )

    def list_region_masks(self, regions: Output):
        return self.add("ETN_ListRegionMasks", 1, regions=regions)

    def attention_mask(self, model: Output, regions: Output):
        return self.add("ETN_AttentionMask", 1, model=model, regions=regions)

    def apply_controlnet(
        self,
        positive: Output,
        negative: Output,
        controlnet: Output,
        image: Output,
        vae: Output,
        strength=1.0,
        range: tuple[float, float] = (0.0, 1.0),
    ):
        return self.add(
            "ControlNetApplyAdvanced",
            2,
            positive=positive,
            negative=negative,
            control_net=controlnet,
            image=image,
            vae=vae,
            strength=strength,
            start_percent=range[0],
            end_percent=range[1],
        )

    def apply_controlnet_inpainting(
        self,
        positive: Output,
        negative: Output,
        controlnet: Output,
        vae: Output,
        image: Output,
        mask: Output,
        strength=1.0,
        range: tuple[float, float] = (0.0, 1.0),
    ):
        return self.add(
            "ControlNetInpaintingAliMamaApply",
            2,
            positive=positive,
            negative=negative,
            control_net=controlnet,
            vae=vae,
            image=image,
            mask=mask,
            strength=strength,
            start_percent=range[0],
            end_percent=range[1],
        )

    def set_controlnet_type(self, controlnet: Output, mode: ControlMode):
        match mode:
            case ControlMode.pose:
                type = "openpose"
            case ControlMode.depth | ControlMode.hands:
                type = "depth"
            case ControlMode.scribble | ControlMode.soft_edge:
                type = "hed/pidi/scribble/ted"
            case ControlMode.line_art | ControlMode.canny_edge:
                type = "canny/lineart/anime_lineart/mlsd"
            case ControlMode.normal:
                type = "normal"
            case ControlMode.segmentation:
                type = "segment"
            case ControlMode.blur:
                type = "tile"
            case _:
                type = "auto"
        return self.add("SetUnionControlNetType", 1, control_net=controlnet, type=type)

    def encode_clip_vision(self, clip_vision: Output, image: Output):
        return self.add("CLIPVisionEncode", 1, clip_vision=clip_vision, image=image)

    def apply_style_model(self, conditioning: Output, style_model: Output, embeddings: Output):
        return self.add(
            "StyleModelApply",
            1,
            conditioning=conditioning,
            style_model=style_model,
            clip_vision_output=embeddings,
        )

    def encode_ip_adapter(
        self, image: Output, weight: float, ip_adapter: Output, clip_vision: Output
    ):
        return self.add(
            "IPAdapterEncoder",
            2,
            image=image,
            weight=weight,
            ipadapter=ip_adapter,
            clip_vision=clip_vision,
        )

    def combine_ip_adapter_embeds(self, embeds: list[Output]):
        e = {f"embed{i + 1}": embed for i, embed in enumerate(embeds)}
        return self.add("IPAdapterCombineEmbeds", 1, method="concat", **e)

    def apply_ip_adapter(
        self,
        model: Output,
        ip_adapter: Output,
        clip_vision: Output,
        embeds: Output,
        weight: float,
        weight_type: str = "linear",
        range: tuple[float, float] = (0.0, 1.0),
        mask: Output | None = None,
    ):
        return self.add(
            "IPAdapterEmbeds",
            1,
            model=model,
            ipadapter=ip_adapter,
            pos_embed=embeds,
            clip_vision=clip_vision,
            weight=weight,
            weight_type=weight_type,
            embeds_scaling="V only",
            start_at=range[0],
            end_at=range[1],
            attn_mask=mask,
        )

    def apply_ip_adapter_face(
        self,
        model: Output,
        ip_adapter: Output,
        clip_vision: Output,
        insightface: Output,
        image: Output,
        weight=1.0,
        range: tuple[float, float] = (0.0, 1.0),
        mask: Output | None = None,
    ):
        return self.add(
            "IPAdapterFaceID",
            1,
            model=model,
            ipadapter=ip_adapter,
            image=image,
            clip_vision=clip_vision,
            insightface=insightface,
            weight=weight,
            weight_faceidv2=weight * 2,
            weight_type="linear",
            combine_embeds="concat",
            start_at=range[0],
            end_at=range[1],
            attn_mask=mask,
            embeds_scaling="V only",
        )

    def apply_self_attention_guidance(self, model: Output):
        return self.add("SelfAttentionGuidance", 1, model=model, scale=0.5, blur_sigma=2.0)

    def inpaint_preprocessor(self, image: Output, mask: Output, fill_black=False):
        return self.add(
            "InpaintPreprocessor", 1, image=image, mask=mask, black_pixel_for_xinsir_cn=fill_black
        )

    def apply_fooocus_inpaint(self, model: Output, patch: Output, latent: Output):
        return self.add("INPAINT_ApplyFooocusInpaint", 1, model=model, patch=patch, latent=latent)

    def vae_encode_inpaint_conditioning(
        self, vae: Output, image: Output, mask: Output, positive: Output, negative: Output
    ):
        return self.add(
            "INPAINT_VAEEncodeInpaintConditioning",
            4,
            vae=vae,
            pixels=image,
            mask=mask,
            positive=positive,
            negative=negative,
        )

    def vae_encode(self, vae: Output, image: Output):
        return self.add("VAEEncode", 1, vae=vae, pixels=image)

    def vae_encode_inpaint(self, vae: Output, image: Output, mask: Output):
        return self.add("VAEEncodeForInpaint", 1, vae=vae, pixels=image, mask=mask, grow_mask_by=0)

    def vae_encode_tiled(self, vae: Output, image: Output):
        return self.add("VAEEncodeTiled", 1, vae=vae, pixels=image, tile_size=512, overlap=64)

    def vae_decode(self, vae: Output, latent_image: Output):
        return self.add("VAEDecode", 1, vae=vae, samples=latent_image)

    def vae_decode_tiled(self, vae: Output, latent_image: Output):
        return self.add(
            "VAEDecodeTiled", 1, vae=vae, samples=latent_image, tile_size=512, overlap=64
        )

    def set_latent_noise_mask(self, latent: Output, mask: Output):
        return self.add("SetLatentNoiseMask", 1, samples=latent, mask=mask)

    def batch_latent(self, latent: Output, batch_size: int):
        if batch_size == 1:
            return latent
        return self.add("RepeatLatentBatch", 1, samples=latent, amount=batch_size)

    def crop_latent(self, latent: Output, bounds: Bounds):
        return self.add(
            "LatentCrop",
            1,
            samples=latent,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def empty_image(self, extent: Extent, color=0):
        return self.add(
            "EmptyImage", 1, width=extent.width, height=extent.height, color=color, batch_size=1
        )

    def crop_image(self, image: Output, bounds: Bounds):
        return self.add(
            "ImageCrop",
            1,
            image=image,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def scale_image(self, image: Output, extent: Extent, method="lanczos"):
        return self.add(
            "ImageScale",
            1,
            image=image,
            width=extent.width,
            height=extent.height,
            upscale_method=method,
            crop="disabled",
        )

    def scale_control_image(self, image: Output, extent: Extent):
        return self.add(
            "HintImageEnchance",
            1,
            hint_image=image,
            image_gen_width=extent.width,
            image_gen_height=extent.height,
            resize_mode="Just Resize",
        )

    def upscale_image(self, upscale_model: Output, image: Output):
        self.sample_count += 4  # approx, actual number depends on model and image size
        return self.add("ImageUpscaleWithModel", 1, upscale_model=upscale_model, image=image)

    def invert_image(self, image: Output):
        return self.add("ImageInvert", 1, image=image)

    def batch_image(self, batch: Output, image: Output):
        return self.add("ImageBatch", 1, image1=batch, image2=image)

    def image_batch_element(self, batch: Output, index: int):
        return self.add("ImageFromBatch", 1, image=batch, batch_index=index, length=1)

    def inpaint_image(self, model: Output, image: Output, mask: Output):
        return self.add(
            "INPAINT_InpaintWithModel", 1, inpaint_model=model, image=image, mask=mask, seed=834729
        )

    def crop_mask(self, mask: Output, bounds: Bounds):
        return self.add(
            "CropMask",
            1,
            mask=mask,
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )

    def scale_mask(self, mask: Output, extent: Extent):
        img = self.mask_to_image(mask)
        scaled = self.scale_image(img, extent, method="bilinear")
        return self.image_to_mask(scaled)

    def image_to_mask(self, image: Output):
        return self.add("ImageToMask", 1, image=image, channel="red")

    def composite_image_masked(
        self, source: Output, destination: Output, mask: Output | None, x=0, y=0
    ):
        return self.add(
            "ImageCompositeMasked",
            1,
            source=source,
            destination=destination,
            mask=mask,
            x=x,
            y=y,
            resize_source=False,
        )

    def mask_to_image(self, mask: Output):
        return self.add("MaskToImage", 1, mask=mask)

    def batch_mask(self, batch: Output, mask: Output):
        image_batch = self.mask_to_image(batch)
        image = self.mask_to_image(mask)
        return self.image_to_mask(self.batch_image(image_batch, image))

    def mask_batch_element(self, mask_batch: Output, index: int):
        image_batch = self.mask_to_image(mask_batch)
        image = self.image_batch_element(image_batch, index)
        return self.image_to_mask(image)

    def solid_mask(self, extent: Extent, value=1.0):
        return self.add("SolidMask", 1, width=extent.width, height=extent.height, value=value)

    def fill_masked(self, image: Output, mask: Output, mode="neutral", falloff: int = 0):
        return self.add("INPAINT_MaskedFill", 1, image=image, mask=mask, fill=mode, falloff=falloff)

    def blur_masked(self, image: Output, mask: Output, blur: int, falloff: int = 0):
        return self.add("INPAINT_MaskedBlur", 1, image=image, mask=mask, blur=blur, falloff=falloff)

    def expand_mask(self, mask: Output, grow: int, blur: int):
        return self.add("INPAINT_ExpandMask", 1, mask=mask, grow=grow, blur=blur)

    def denoise_to_compositing_mask(self, mask: Output, offset=0.05, threshold=0.35):
        return self.add(
            "INPAINT_DenoiseToCompositingMask", 1, mask=mask, offset=offset, threshold=threshold
        )

    def apply_mask(self, image: Output, mask: Output):
        return self.add("ETN_ApplyMaskToImage", 1, image=image, mask=mask)

    def translate(self, text: str | Output):
        return self.add("ETN_Translate", 1, text=text)

    def nsfw_filter(self, image: Output, sensitivity: float):
        if sensitivity > 0:
            return self.add("ETN_NSFWFilter", 1, image=image, sensitivity=sensitivity)
        return image

    def load_image(self, image: Image | ImageCollection):
        if self._run_mode is ComfyRunMode.runtime:
            return self._load_image_batch(image, self._load_image_runtime, self.batch_image)
        else:
            return self._load_image_batch(image, self._load_image_base64, self.batch_image)

    def load_mask(self, mask: Image | ImageCollection):
        if self._run_mode is ComfyRunMode.runtime:
            return self._load_image_batch(mask, self._load_mask_runtime, self.batch_mask)
        else:
            return self._load_image_batch(mask, self._load_mask_base64, self.batch_mask)

    def _load_image_runtime(self, image: Image):
        return self.add("ETN_InjectImage", 1, id=self._add_image(image))

    def _load_image_base64(self, image: Image):
        return self.add("ETN_LoadImageBase64", 1, image=image.to_base64())

    def _load_mask_runtime(self, mask: Image):
        return self.add("ETN_InjectMask", 1, id=self._add_image(mask))

    def _load_mask_base64(self, mask: Image):
        return self.add("ETN_LoadMaskBase64", 1, mask=mask.to_base64())

    def _load_image_batch(self, images: Image | ImageCollection, loader, batcher) -> Output:
        if isinstance(images, Image):
            return loader(images)
        result = None
        for image in images:
            img = loader(image)
            result = img if result is None else batcher(result, img)
        assert result is not None
        return result

    def load_image_and_mask(self, images: Image | ImageCollection):
        assert self._run_mode is ComfyRunMode.server
        if isinstance(images, Image):
            return self.add("ETN_LoadImageBase64", 2, image=images.to_base64())
        result = None
        for image in images:
            img, mask = self.add("ETN_LoadImageBase64", 2, image=image.to_base64())
            if result:
                result = (self.batch_image(result[0], img), self.batch_mask(result[1], mask))
            else:
                result = (img, mask)
        assert result is not None
        return result

    def send_image(self, image: Output):
        if self._run_mode is ComfyRunMode.runtime:
            return self.add("ETN_ReturnImage", 1, images=image)
        return self.add("ETN_SendImageWebSocket", 1, images=image, format="PNG")

    def save_image(self, image: Output, prefix: str):
        return self.add("SaveImage", 1, images=image, filename_prefix=prefix)

    def create_tile_layout(self, image: Output, tile_size: int, padding: int, blending: int):
        return self.add(
            "ETN_TileLayout",
            1,
            image=image,
            min_tile_size=tile_size,
            padding=padding,
            blending=blending,
        )

    def extract_image_tile(self, image: Output, layout: Output, index: int):
        return self.add("ETN_ExtractImageTile", 1, image=image, layout=layout, index=index)

    def extract_mask_tile(self, mask: Output, layout: Output, index: int):
        return self.add("ETN_ExtractMaskTile", 1, mask=mask, layout=layout, index=index)

    def merge_image_tile(self, image: Output, layout: Output, index: int, tile: Output):
        return self.add("ETN_MergeImageTile", 1, layout=layout, index=index, tile=tile, image=image)

    def generate_tile_mask(self, layout: Output, index: int):
        return self.add("ETN_GenerateTileMask", 1, layout=layout, index=index)

    def define_reference_image(
        self, references: Output | None, image: Output, weight: float, range: tuple[float, float]
    ):
        return self.add(
            "ETN_ReferenceImage",
            1,
            image=image,
            weight=weight,
            range_start=range[0],
            range_end=range[1],
            reference_images=references,
        )

    def apply_reference_images(
        self, conditioning: Output, clip_vision: Output, style_model: Output, references: Output
    ):
        return self.add(
            "ETN_ApplyReferenceImages",
            1,
            conditioning=conditioning,
            clip_vision=clip_vision,
            style_model=style_model,
            references=references,
        )

    def estimate_pose(self, image: Output, resolution: int):
        feat = dict(detect_hand="enable", detect_body="enable", detect_face="enable")
        mdls = dict(bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384.onnx")
        if self._run_mode is ComfyRunMode.runtime:
            # use smaller model, but it requires onnxruntime, see #630
            mdls["bbox_detector"] = "yolo_nas_l_fp16.onnx"
        return self.add("DWPreprocessor", 1, image=image, resolution=resolution, **feat, **mdls)

    def apply_first_block_cache(self, model: Output, arch: Arch):
        return self.add(
            "ApplyFBCacheOnModel",
            1,
            model=model,
            object_to_patch="diffusion_model",
            residual_diff_threshold=0.2 if arch.is_sdxl_like else 0.12,
            start=0.0,
            end=1.0,
            max_consecutive_cache_hits=-1,
        )

    def create_hook_lora(self, loras: list[tuple[str, float]]):
        key = "CreateHookLora" + str(loras)
        hooks = self._cache.get(key, None)
        if hooks is None:
            for lora, strength in loras:
                hooks = self.add(
                    "CreateHookLora",
                    1,
                    lora_name=lora,
                    strength_model=strength,
                    strength_clip=strength,
                    prev_hooks=hooks,
                )
            assert hooks is not None
            self._cache[key] = hooks

        assert isinstance(hooks, Output)
        return hooks

    def set_clip_hooks(self, clip: Output, hooks: Output):
        return self.add(
            "SetClipHooks", 1, clip=clip, hooks=hooks, apply_to_conds=True, schedule_clip=False
        )

    def combine_masked_conditioning(
        self,
        positive: Output,
        negative: Output,
        positive_conds: Output | None = None,
        negative_conds: Output | None = None,
        mask: Output | None = None,
    ):
        assert (positive_conds and negative_conds) or mask
        if mask is None:
            return self.add(
                "PairConditioningSetDefaultCombine",
                2,
                positive=positive_conds,
                negative=negative_conds,
                positive_DEFAULT=positive,
                negative_DEFAULT=negative,
            )
        if positive_conds is None and negative_conds is None:
            return self.add(
                "PairConditioningSetProperties",
                2,
                positive_NEW=positive,
                negative_NEW=negative,
                mask=mask,
                strength=1.0,
                set_cond_area="default",
            )
        return self.add(
            "PairConditioningSetPropertiesAndCombine",
            2,
            positive=positive_conds,
            negative=negative_conds,
            positive_NEW=positive,
            negative_NEW=negative,
            mask=mask,
            strength=1.0,
            set_cond_area="default",
        )


def _inputs_for_node(node_inputs: dict[str, dict[str, Any]], node_name: str, filter=""):
    inputs = node_inputs.get(node_name)
    if inputs is None:
        return None
    if filter:
        return inputs.get(filter)
    result = inputs.get("required", {})
    result.update(inputs.get("optional", {}))
    return result


def _convert_ui_workflow(w: dict, node_inputs: dict):
    version = w.get("version")
    nodes = w.get("nodes")
    links = w.get("links")
    if not (version and nodes and links):
        return w

    if not node_inputs:
        raise ValueError("An active ComfyUI connection is required to convert a UI workflow file.")

    primitives = {}
    for node in nodes:
        if node["type"] == "PrimitiveNode":
            primitives[node["id"]] = node["widgets_values"][0]

    r = {}
    for node in nodes:
        id = node["id"]
        type = node["type"]
        if type == "PrimitiveNode":
            continue

        inputs = {}
        fields = _inputs_for_node(node_inputs, type)
        if fields is None:
            raise ValueError(
                f"Workflow uses node type {type}, but it is not installed on the ComfyUI server."
            )
        widget_count = 0
        for field_name, field in fields.items():
            field_type = field[0]
            if field_type in ["INT", "FLOAT", "BOOL", "STRING"] or isinstance(field_type, list):
                values = node["widgets_values"]
                inputs[field_name] = values[widget_count]
                widget_count += 1
                if len(values) > widget_count and values[widget_count] in _control_after_generate:
                    widget_count += 1

            for connection in node["inputs"]:
                if connection["name"] == field_name and connection["link"] is not None:
                    link = next(x for x in links if x[0] == connection["link"])
                    prim = primitives.get(link[1])
                    if prim is not None:
                        inputs[field_name] = prim
                    else:
                        inputs[field_name] = [link[1], link[2]]
                    break
        r[id] = {"class_type": type, "inputs": inputs}

    return r


_control_after_generate = ["fixed", "increment", "decrement", "randomize"]
