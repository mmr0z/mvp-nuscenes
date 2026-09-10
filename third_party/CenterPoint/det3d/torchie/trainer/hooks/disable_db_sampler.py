from .hook import Hook


class DisableDBSamplerHook(Hook):
    def __init__(self, disable_after_epoch=12):
        self.disable_after_epoch = disable_after_epoch
        self._disabled = False

    def before_train_epoch(self, trainer):
        if self._disabled or trainer.epoch < self.disable_after_epoch:
            return

        dataset = trainer.data_loader.dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset

        pipeline = getattr(dataset, "pipeline", None)
        transforms = getattr(pipeline, "transforms", [])

        disabled = 0
        for transform in transforms:
            if hasattr(transform, "db_sampler") and transform.db_sampler is not None:
                transform.db_sampler = None
                disabled += 1

        trainer.logger.info(
            "DisableDBSamplerHook disabled db_sampler in %d transforms at epoch %d",
            disabled,
            trainer.epoch + 1,
        )
        self._disabled = True
