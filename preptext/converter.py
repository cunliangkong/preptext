# -*- coding: utf-8 -*-
import torch
from .dataset import Dataset, BucketSampler, BucketCollate


def to_dataset(datastorage, key=None, reverse=False):
    dataset = Dataset(datastorage, key=key, reverse=reverse)
    return dataset


def to_dataloader(datastorage,
                  batch_size=1,
                  shuffle=False,
                  num_workers=0,
                  pin_memory=False,
                  drop_last=False,
                  timeout=0,
                  worker_init_fn=None):
    dataset = to_dataset(datastorage)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)
    return dataloader


def to_bucketdataloader(datastorage,
                        key=None,
                        reverse=False,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False,
                        drop_last=False,
                        timeout=0,
                        worker_init_fn=None):
    dataset = to_dataset(datastorage, key=key, reverse=reverse)
    bucket_sampler = BucketSampler(dataset,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   drop_last=drop_last)
    bucket_collate = BucketCollate(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_sampler=bucket_sampler,
                                             num_workers=num_workers,
                                             collate_fn=bucket_collate,
                                             pin_memory=pin_memory,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)
    return dataloader


def to_distributeddataloader(datastorage,
                             world_size,
                             rank,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=False,
                             timeout=0,
                             worker_init_fn=None):
    dataset = to_dataset(datastorage)
    distributed_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)
    distributed_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=distributed_sampler,
        timeout=timeout,
        worker_init_fn=worker_init_fn)
    return distributed_loader
