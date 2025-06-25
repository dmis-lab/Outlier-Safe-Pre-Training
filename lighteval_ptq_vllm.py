from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

from eval_utils.main import ptq_model
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"


def create_ptq_args(
    w_bits: int = 16,
    a_bits: int = 16,
    kv_bits: int = 16,
    rotate: bool = False,
    rotate_down_proj: bool = False,
    no_rtn: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        rotate=rotate,
        rotate_down_proj=rotate_down_proj,
        rotate_mode="hadamard",
        rotation_seed=-1,
        fp32_had=False,
        nsamples=128,
        percdamp=0.01,
        act_order=False,
        int8_down_proj=False,
        load_qmodel_path=None,
        save_qmodel_path=None,
        export_to_et=False,
        capture_layer_io=False,
        layer_idx=10,
        # Activation
        a_bits=a_bits,
        a_groupsize=-1,
        a_asym=True,
        a_clip_ratio=1.0,
        # Weights
        w_bits=w_bits,
        w_groupsize=-1,
        w_asym=False,
        w_rtn=not no_rtn,
        w_clip=True,
        # Key
        k_bits=kv_bits,
        k_groupsize=128,
        k_asym=True,
        k_pre_rope=False,
        k_clip_ratio=1.0,
        # Value
        v_bits=kv_bits,
        v_groupsize=128,
        v_asym=True,
        v_clip_ratio=1.0,
    )


def main(args: argparse.Namespace):
    ptq_args = create_ptq_args(
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        kv_bits=16,
        rotate=args.rotate,
        rotate_down_proj=args.rotate_down_proj,
        no_rtn=args.no_rtn,
    )
    model_args = SimpleNamespace(input_model=args.pretrained)

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=args.save_details,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory=args.custom_tasks,
    )
    model_config = VLLMModelConfig(
        model_name=args.pretrained,
        dtype=args.dtype,
        max_model_length=args.max_model_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    pipeline = Pipeline(
        tasks=args.tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    model = pipeline.model.model
    model = model.llm_engine.model_executor.driver_worker.model_runner.model
    model = ptq_model(ptq_args, model, model_args).cuda()

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_length", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--tasks", default="tasks.txt")
    parser.add_argument("--custom_tasks", default="lighteval_tasks.py")
    parser.add_argument("--save_details", action="store_true", default=False)
    parser.add_argument("--w_bits", type=int, default=16)
    parser.add_argument("--a_bits", type=int, default=16)
    parser.add_argument("--rotate", action="store_true", default=False)
    parser.add_argument("--rotate_down_proj", action="store_true", default=False)
    parser.add_argument("--no_rtn", action="store_true", default=False)
    main(parser.parse_args())
