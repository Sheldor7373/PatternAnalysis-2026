"""Bind inference to actual artifacts and a reproducible effective runtime."""
import hashlib
import json
import os
from pathlib import Path
import platform

import torch
from transformers import AutoConfig


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def artifact_manifest(directory):
    root=Path(directory).resolve()
    paths={path.name:path for path in root.iterdir() if path.is_file()
           and path.suffix in {'.json','.model','.safetensors','.bin','.txt'}}
    # Transformers shard indices may refer to nested files. Include the actual
    # referenced shards as well as ordinary top-level .bin/safetensors weights.
    for name,path in list(paths.items()):
        if name.endswith('.index.json'):
            index=json.loads(path.read_text(encoding='utf-8'))
            for relative in index.get('weight_map',{}).values():
                part=Path(relative)
                if part.is_absolute() or '..' in part.parts:
                    raise ValueError('Model shard index leaves the artifact directory')
                target=root/part
                if not target.is_file():
                    raise ValueError(f'Model shard is missing: {relative}')
                paths[part.as_posix()]=target
    if not any(Path(name).suffix in {'.bin','.safetensors'} for name in paths):
        raise ValueError('Local model artifact contains no supported weights')
    return {name:digest(path) for name,path in sorted(paths.items())}


def bind_source(name,revision):
    local=Path(name)
    if local.is_dir():
        location=str(local.resolve())
        return {'source':location,'revision':revision,
                'identity':{'kind':'local','files_sha256':artifact_manifest(location)}}
    configuration=AutoConfig.from_pretrained(name,revision=revision)
    pinned=getattr(configuration,'_commit_hash',None)
    if not pinned or pinned in {'main','master'}:
        raise ValueError('Remote model/tokenizer source did not resolve to an immutable revision')
    return {'source':name,'revision':pinned,
            'identity':{'kind':'huggingface','model':name,'model_revision':pinned,
                        'tokenizer_revision':pinned}}


def verify_source(binding):
    if binding['identity']['kind']=='local':
        if artifact_manifest(binding['source'])!=binding['identity']['files_sha256']:
            raise ValueError('Model source changed while it was loading')


def resolve_precision(requested,device_type,bf16_available=False):
    if requested not in {'auto','bf16','fp32'}:
        raise ValueError('Unknown precision request')
    if requested=='bf16':
        if device_type!='cuda' or not bf16_available:
            raise ValueError('bf16 requested on unsupported device')
        return 'bf16'
    if requested=='auto' and device_type=='cuda' and bf16_available:
        return 'bf16'
    return 'fp32'


def effective_runtime(requested,device):
    available=device.type=='cuda' and torch.cuda.is_bf16_supported()
    result={'device_type':device.type,
            'effective_precision':resolve_precision(requested,device.type,available),
            'cpu_threads':torch.get_num_threads(),'cpu_interop_threads':torch.get_num_interop_threads(),
            'python':platform.python_version(),'machine':platform.machine(),
            'tokenizers_parallelism':os.environ.get('TOKENIZERS_PARALLELISM'),
            'rayon_threads':os.environ.get('RAYON_NUM_THREADS'),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'float32_matmul_precision':torch.get_float32_matmul_precision()}
    if device.type=='cuda':
        index=device.index if device.index is not None else torch.cuda.current_device()
        properties=torch.cuda.get_device_properties(index)
        result.update(device_index=index,gpu_name=properties.name,
                      compute_capability=[properties.major,properties.minor],
                      total_memory_bytes=properties.total_memory,cuda_runtime=torch.version.cuda,
                      allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                      allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
    return result
