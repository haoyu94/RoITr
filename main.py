import copy
import os, argparse, json, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # for Pytorch DistributedDataParallel(DDP) training
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler # for Pytorch DistbutedDataParallel(DDP) training
from lib.utils import setup_seed
from configs.utils import load_config
from easydict import EasyDict as edict
from dataset.dataloader import get_dataset, get_dataloader
from model.RIGA_v2 import create_model
from lib.loss import OverallLoss, Evaluator
from lib.tester import get_trainer


def main():
    #########################################################
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--local_rank", type=int, default=-1) # for DDP training
    args = parser.parse_args()
    config = load_config(args.config)
    config['local_rank'] = args.local_rank
    #########################################################
    #set cuda devices for both DDP training and single-GPU training
    if config['local_rank'] > -1:
        torch.cuda.set_device(config['local_rank'])
        config['device'] = torch.device('cuda', config['local_rank'])
        torch.distributed.init_process_group(backend='nccl')

    else:
        torch.cuda.set_device(0)
        config['device'] = torch.device('cuda', 0)

    ##########################################################
    setup_seed(42) # fix the seed

    ##########################################################
    # set paths for storing results
    config['snapshot_dir'] = 'snapshot/{}'.format(config['exp_dir'])
    config['tboard_dir'] = 'snapshot/{}/tensorboard'.format(config['exp_dir'])
    config['save_dir'] = 'snapshot/{}/checkpoints'.format(config['exp_dir'])
    config['visual_dir'] = 'snapshot/{}/visualization'.format(config['exp_dir'])
    ##########################################################
    config = edict(config)

    ################################################################
    # backup files and configurations
    if config.local_rank <= 0:
        os.makedirs(config.snapshot_dir, exist_ok=True)
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.tboard_dir, exist_ok=True)
        os.makedirs(config.visual_dir, exist_ok=True)
        config_ = copy.deepcopy(config)
        config_.device = config.local_rank
        json.dump(
            config_,
            open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
            indent=4,
        )
        os.system(f'cp -r model {config.snapshot_dir}')
        os.system(f'cp -r dataset {config.snapshot_dir}')
        os.system(f'cp -r lib {config.snapshot_dir}')
        shutil.copy2('main.py', config.snapshot_dir)
    ##################################################################
    # create model
    #config.model = RIGA(config=config).to(config.device)
    config.model = create_model(config).to(config.device)

    # print the details of network architecture
    if config.local_rank <= 0:
        print(config.model)
    # for PyTorch DistbutedDataParallel(DDP) training
    if config.local_rank >= 0:
        config.model = torch.nn.parallel.DistributedDataParallel(config.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)


    # create optimizer
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    # create scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )

    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_dataset(config)
    if config.local_rank > -1:
        train_sampler, val_sampler, benchmark_sampler = DistributedSampler(train_set), DistributedSampler(val_set), DistributedSampler(benchmark_set)
    else:
        train_sampler = val_sampler = benchmark_sampler = None

    config.train_loader = get_dataloader(train_set,
                                         sampler=train_sampler,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         shuffle=True,
                                         drop_last=True)
    config.val_loader = get_dataloader(val_set,
                                       sampler=val_sampler,
                                       batch_size=1,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       drop_last=False)
    config.test_loader = get_dataloader(benchmark_set,
                                        sampler=benchmark_sampler,
                                        batch_size=1,
                                        num_workers=config.num_workers,
                                        shuffle=False,
                                        drop_last=False)
    # create losses and evaluation metrics
    config.loss_func = OverallLoss(config)
    config.evaluator = Evaluator(config)
    trainer = get_trainer(config)
    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'val':
        trainer.eval()
    elif config.mode == 'test':
        trainer.test()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
