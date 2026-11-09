import os
import utils
import shutil
import logging
import argparse
import importlib
import torch
import torch.distributed as dist
from datetime import datetime
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import EpochBasedRunner, build_optimizer, load_checkpoint
from mmdet.apis import set_random_seed
from mmdet.core import DistEvalHook, EvalHook
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from loaders.builder import build_dataloader
from loaders.apis.test import DistEvalHookConcatFast


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--work-dir')
    parser.add_argument('--benchmark-warmup', type=int, default=20)
    parser.add_argument('--benchmark-iters', type=int, default=0)
    parser.add_argument('--benchmark-output')
    parser.add_argument('--batch-size-per-gpu', type=int)
    parser.add_argument('--stop-after-epochs', type=int)
    args = parser.parse_args()
    if args.benchmark_iters < 0 or args.benchmark_warmup < 0:
        parser.error('Benchmark iteration counts must be nonnegative.')
    if args.batch_size_per_gpu is not None and args.batch_size_per_gpu < 1:
        parser.error('Batch size must be positive.')
    if args.stop_after_epochs is not None and args.stop_after_epochs < 1:
        parser.error('Epoch limit must be positive.')

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmdet3d'] = logging.Logger(__name__, logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if args.batch_size_per_gpu is not None:
        cfgs.per_gpu_batch_size = args.batch_size_per_gpu
    cfgs.batch_size = cfgs.per_gpu_batch_size * world_size

    if local_rank == 0:
        # resume or start a new run
        if cfgs.resume_from is not None:
            assert os.path.isfile(cfgs.resume_from)
            work_dir = os.path.dirname(cfgs.resume_from)
        else:
            run_name = ''
            # if not cfgs.debug:
            #     run_name = input('Name your run (leave blank for default): ')
            if run_name == '':
                run_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S-%f")

            # work_dir = os.path.join('outputs', cfgs.output_dir, run_name)
            work_dir = args.work_dir or os.path.join(
                os.environ.get('STREAMRC_OUTPUT_ROOT', 'outputs'), cfgs.output_dir, run_name)

            os.makedirs(work_dir, exist_ok=False)

        # init logging, backup code
        utils.init_logging(os.path.join(work_dir, 'train.log'), cfgs.debug)
        utils.backup_code(work_dir)
        logging.info('Logs will be saved to %s' % work_dir)

    else:
        # disable logging on other workers
        logging.root.disabled = True
        work_dir = '/tmp'

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist.init_process_group('nccl', init_method='env://')

    from hpc3_runtime import configure_focal_loss
    focal_implementation = configure_focal_loss(cfgs)

    logging.info('Setting random seed: %d' % cfgs.seed)
    set_random_seed(cfgs.seed, deterministic=True)

    logging.info('Loading training set from %s' % cfgs.dataset_root)
    train_dataset = build_dataset(cfgs.data.train)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfgs.per_gpu_batch_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=cfgs.train_dataloader_shuffle,
        seed=cfgs.seed,
        train=True,
        use_streaming=cfgs.use_streaming,
        **cfgs.get('loader_options', {}),
    )
    if args.batch_size_per_gpu is not None or cfgs.get('lr_warmup_epochs') is not None:
        # Keep the original approximately one-epoch LR warmup after changing batch.
        cfgs.sample_per_epoch = len(train_loader)
        cfgs.lr_config.warmup_iters = max(1, int(cfgs.get('lr_warmup_epochs', 1) * len(train_loader)))
        for hook_cfg in cfgs.log_config.hooks:
            if hook_cfg.type == 'MyTensorboardLoggerHook':
                hook_cfg.interval = len(train_loader)
        logging.info('Global batch: %d; iterations/epoch and LR warmup: %d',
                     cfgs.batch_size, len(train_loader))

    if not args.benchmark_iters:
        logging.info('Loading validation set from %s' % cfgs.dataset_root)
        val_dataset = build_dataset(cfgs.data.val)
        val_batch_size = cfgs.get('val_batch_size_per_gpu', 1) if world_size > 1 else 1
        if val_batch_size < 1:
            raise ValueError('Validation batch size per GPU must be positive')
        logging.info('Validation batch size per GPU: %d', val_batch_size)
        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_batch_size,
            workers_per_gpu=cfgs.data.workers_per_gpu,
            num_gpus=world_size,
            dist=world_size > 1,
            shuffle=False,
            train=False,
            seed=cfgs.seed,
            use_streaming=True,
            **cfgs.get('loader_options', {}),
        )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.init_weights()
    model.cuda()
    model.train()
    from hpc3_speed import configure_training_speed
    speed_settings = configure_training_speed(model, cfgs.get('hpc3_speed', {}))

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info('Trainable parameters: %d (%.1fM)' % (n_params, n_params / 1e6))
    logging.info('Batch size per GPU: %d' % cfgs.per_gpu_batch_size)

    if world_size > 1:
        model = MMDistributedDataParallel(model, [local_rank], broadcast_buffers=False, find_unused_parameters=True)
    else:
        model = MMDataParallel(model, [0])

    logging.info('Creating optimizer: %s' % cfgs.optimizer.type)
    if speed_settings.get('enabled', False):
        cfgs.optimizer.fused = speed_settings.get('fused_optimizer', True)
    optimizer = build_optimizer(model, cfgs.optimizer)
    from hpc3_speed import configure_optimizer_speed
    configure_optimizer_speed(optimizer, speed_settings)

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=work_dir,
        logger=logging.root,
        max_epochs=cfgs.total_epochs,
        meta=dict(),
    )

    runner.register_lr_hook(cfgs.lr_config)
    runner.register_optimizer_hook(cfgs.optimizer_config)
    runner.register_checkpoint_hook(cfgs.checkpoint_config)
    runner.register_logger_hooks(cfgs.log_config)
    runner.register_timer_hook(dict(type='IterTimerHook'))
    runner.register_custom_hooks(dict(type='DistSamplerSeedHook'))

    if not args.benchmark_iters and cfgs.eval_config['interval'] > 0:
        if world_size > 1:
            # runner.register_hook(DistEvalHook(val_loader, interval=cfgs.eval_config['interval'], gpu_collect=True))
            runner.register_hook(DistEvalHookConcatFast(val_loader, gpu_collect=True, **dict(cfgs.eval_config)))
        else:
            eval_options = dict(cfgs.eval_config)
            eval_options.pop('batch_search', None)
            runner.register_hook(EvalHook(val_loader, **eval_options))

    if cfgs.resume_from is not None:
        logging.info('Resuming from %s' % cfgs.resume_from)
        runner.resume(cfgs.resume_from)

    elif cfgs.load_from is not None:
        logging.info('Loading checkpoint from %s' % cfgs.load_from)
        if cfgs.revise_keys is not None:
            load_checkpoint(
                model, cfgs.load_from, map_location='cpu',
                revise_keys=cfgs.revise_keys
            )
        else:
            load_checkpoint(
                model, cfgs.load_from, map_location='cpu',
            )
  
    if args.benchmark_iters:
        from benchmark_runtime import BenchmarkComplete, ThroughputHook
        from models.csrc.wrapper import MSMV_CUDA
        output = args.benchmark_output or os.path.join(work_dir, 'throughput.json')
        metadata = dict(
            config=os.path.abspath(args.config), data_root=cfgs.dataset_root,
            training_samples=len(train_dataset),
            batch_size_per_gpu=cfgs.per_gpu_batch_size,
            workers_per_gpu=cfgs.data.workers_per_gpu,
            backbone_depth=cfgs.model.img_backbone.depth,
            optimizer=dict(type=cfgs.optimizer.type, lr=cfgs.optimizer.lr),
            optimizer_hook=dict(cfgs.optimizer_config),
            msmv_implementation='custom_cuda' if MSMV_CUDA else 'torch_grid_sample',
            focal_implementation=focal_implementation,
            execution_settings=speed_settings,
            torch_version=torch.__version__,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        )
        runner.register_hook(ThroughputHook(
            args.benchmark_warmup, args.benchmark_iters, output, metadata), priority='LOWEST')
        try:
            runner.run([train_loader], [('train', 1)])
        except BenchmarkComplete:
            runner.call_hook('after_run')
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    else:
        from training_health import TrainingHealthHook
        runner.register_hook(TrainingHealthHook(), priority='HIGH')
        from epoch_timing import EpochTimingHook, EpochEndTimingHook, TrainingLimitReached
        timing = EpochTimingHook(dict(
            config=os.path.abspath(args.config), batch_size_per_gpu=cfgs.per_gpu_batch_size,
            global_batch_size=cfgs.batch_size, training_samples=len(train_dataset),
            iterations_per_epoch=len(train_loader), total_scheduled_epochs=cfgs.total_epochs,
            execution_settings=speed_settings,
            lr_warmup_iterations=cfgs.lr_config.warmup_iters,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        ))
        runner.register_hook(timing, priority='ABOVE_NORMAL')
        runner.register_hook(EpochEndTimingHook(timing, args.stop_after_epochs), priority='LOWEST')
        try:
            runner.run([train_loader], [('train', 1)])
        except TrainingLimitReached:
            runner.call_hook('after_run')
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == '__main__':
    main()
