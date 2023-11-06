import argparse

from pytorch_lightning import Trainer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.data_module_joint import FinetuneDataModuleJoint
from src.evaluation.evaluate import Evaluator
from src.evaluation.evaluate_natop import EvaluatorNatop
from src.evaluation.evaluate_verdict import EvaluatorVerdict
from src.lit_module import LitModule
from src.models.tfew import EncoderDecoder
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds


def get_transformer(config):
    # Selected model does not matter when loading from cache. Just filler.
    tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)

    tokenizer.model_max_length = config.max_seq_len
    return tokenizer, model


def main(config):
    """
    Trains the model

    :param config:
    :return:
    """

    assert config.load_weight != "", "Need to specify a model to load..."

    tokenizer, model = get_transformer(config)

    datamodule_natop = FinetuneDataModuleJoint(config, tokenizer, mode="natop")
    datamodule_verdict = FinetuneDataModuleJoint(config, tokenizer, mode="verdict")

    evaluator_natop = EvaluatorNatop(config, datamodule_natop)
    evaluator_verdict = EvaluatorVerdict(config, datamodule_verdict)

    model = EncoderDecoder(config, tokenizer, model)

    litmodule = LitModule(config, model, datamodule_natop, evaluator_natop)
    litmodule_verdict = LitModule(config, model, datamodule_verdict, evaluator_verdict)

    trainer = Trainer(
        enable_checkpointing=False,
        accelerator="gpu",
        devices=1,
        precision=config.compute_precision,
        strategy=config.compute_strategy if config.compute_strategy != "none" else None,
    )

    # Generate natop probability score and verdict score
    trainer.test(litmodule, datamodule_natop)
    trainer.test(litmodule_verdict, datamodule_verdict)

    # Use scores to select the most appropriate proof given the average of both scores
    evaluator = Evaluator(config, datamodule_natop)
    evaluator.run_cached_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)
    print(f"Start experiment {config.exp_name}")
    # Setup config
    assert config.compute_strategy in ["none", "ddp", "deepspeed_stage_3_offload", "deepspeed_stage_3"]

    print(config.to_json())

    set_seeds(config.seed)
    main(config)
