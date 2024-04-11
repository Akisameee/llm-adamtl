import os
# from peft.config import PeftConfig
from transformers.generation.configuration_utils import GenerationConfig
import typing
from typing import Literal, Optional
import json
import numpy as np
import dataclasses
from dataclasses import dataclass, field, asdict
import argparse
import tyro
from typing_extensions import Annotated
from copy import deepcopy

JSONDict = Annotated[Optional[dict], tyro.conf.arg(metavar="JSON", constructor=json.loads)]

@dataclass
class Base_Config(object):

    args_namespace: dict = None

    def to_dict(self):

        return asdict(self)

    def get_dataclass_fields(
        self,
        dataclass,
        default = None,
        prefix = None
    ):  
        
        if default is not None:
            dataclass = type(default)

        branches = []
        fields = []
        
        for field in dataclasses.fields(dataclass):
            name_new = '_'.join([prefix, field.name]) if prefix is not None else field.name
            if dataclasses.is_dataclass(field.type):
                sub_branch, sub_field = self.get_dataclass_fields(
                    field.type,
                    field.default if default is None else getattr(default, field.name),
                    prefix = name_new
                )
                branches.append({
                    'name': field.name,
                    'name_new': name_new,
                    'dataclass': sub_branch['dataclass'],
                    'sub': sub_branch['sub']
                })
                fields += sub_field
            else:
                branches.append({
                    'name': field.name,
                    'name_new': name_new,
                    'dataclass': None,
                    'sub': None
                })
                fields.append((
                    name_new,
                    field.type,
                    field.default if default is None else getattr(default, field.name)
                ))

        return {
                'dataclass': dataclass,
                'sub': branches
            }, fields

    def parse_args_into_dataclass(
        self,
        branches,
        args_namespace
    ):
        if isinstance(self, branches['dataclass']):
            args_dataclass = self
        else:
            args_dataclass = branches['dataclass']()
        
        args = {}
        for arg in branches['sub']:
            if arg['dataclass'] is not None:
                sub_args_dataclass = self.parse_args_into_dataclass(arg, args_namespace)
                setattr(args_dataclass, arg['name'], sub_args_dataclass)
            else:
                args[arg['name']] = args_namespace[arg['name_new']]
                setattr(args_dataclass, arg['name'], args_namespace[arg['name_new']])

        return args_dataclass

    def parse_args(self):

        parser = argparse.ArgumentParser()
        branches, fields = self.get_dataclass_fields(dataclass = type(self))

        for name, ftype, fdefault in fields:
            if typing.get_origin(ftype) == typing.Union:
                literal_args = typing.get_args(typing.get_args(ftype)[0])
                arg_type = type(literal_args[0])
                parser.add_argument(f'--{name}', type=arg_type, choices=literal_args, default=fdefault)
            elif typing.get_origin(ftype) == Literal:
                literal_args = typing.get_args(ftype)
                arg_type = type(literal_args[0])
                parser.add_argument(f'--{name}', type=arg_type, choices=literal_args, default=fdefault)
            else:
                parser.add_argument(f'--{name}', type=ftype, default=fdefault)
        
        args_namespace = parser.parse_args()
        args_dataclass = self.parse_args_into_dataclass(branches, vars(args_namespace))
        self.args_namespace = vars(args_namespace)

        return args_dataclass

    def get_args_info(self):

        args_info_str = 'Args Info:\n'
        if self.args_namespace is not None:
            for key, value in self.args_namespace.items():
                if key != 'args_namespace':
                    args_info_str += f'--{key} {value}\n'

        return args_info_str

@dataclass
class Instruct_Dataset_Config(Base_Config):

    model_name: str = None
    data_path: str = os.path.join('/home', 'smliu', 'datasets', 'instruct', 'sharegpt')
    sub_data_path: list = None
    tokenizer_pretrain_path: str = None
    padding_side: Optional[Literal['left', 'right']] = 'left'
    max_len: int = 512
    prompt_only: bool = False
    tokenize_type: Optional[Literal['prompt_pad', 'prompt_not_pad', 'prompt_response']] = 'prompt_pad'
    remove_chinese: bool = True

@dataclass
class HumanFeedback_Dataset_Config(Base_Config):

    data_path: str = None
    sub_data_path: list = None
    name: str = None
    model_name: str = None
    tokenizer_pretrain_path: str = None
    pad_token_id: int = None
    label_pad_token_id: int = -100
    truncation_side: Optional[Literal['left', 'right']] = 'left'
    max_len: int = None

# generation_config = GenerationConfig(
#     top_k = 0.0,
#     top_p = 1.0,
#     do_sample = True,
#     eos_token_id = 32001,
# )

generation_config = GenerationConfig(
    top_k = 50,
    top_p = 1.0,
    do_sample = True,
    eos_token_id = 50256
)

@dataclass
class Accelertor_Config(Base_Config):

    log_with: Optional[Literal['wandb', 'tensorboard']] = None
    gradient_accumulation_steps: int = 1


def get_argparser(
    dataclass
):
    parser = argparse.ArgumentParser()
    for field in dataclasses.fields(dataclass):
        parser.add_argument(f'--{field.name}', type=field.type, default=field.default)
    parser
    return parser

def get_dataclass_fields(
    dataclass,
    default = None,
    prefix = None
):  
    
    if default is not None:
        dataclass = type(default)

    branches = []
    fields = []
    
    for field in dataclasses.fields(dataclass):
        name_new = '_'.join([prefix, field.name]) if prefix is not None else field.name
        if dataclasses.is_dataclass(field.type):
            sub_branch, sub_field = get_dataclass_fields(
                field.type,
                field.default if default is None else getattr(default, field.name),
                prefix = name_new
            )
            branches.append({
                'name': field.name,
                'name_new': name_new,
                'dataclass': sub_branch['dataclass'],
                'sub': sub_branch['sub']
            })
            fields += sub_field
        else:
            branches.append({
                'name': field.name,
                'name_new': name_new,
                'dataclass': None,
                'sub': None
            })
            fields.append((
                name_new,
                field.type,
                field.default if default is None else getattr(default, field.name)
            ))

    return {
            'dataclass': dataclass,
            'sub': branches
        }, fields

def parse_args_into_dataclass(
    branches,
    args_namespace
):
    args_dataclass = branches['dataclass']()
    args = {}
    for arg in branches['sub']:
        if arg['dataclass'] is not None:
            sub_args_dataclass = parse_args_into_dataclass(arg, args_namespace)
            setattr(args_dataclass, arg['name'], sub_args_dataclass)
        else:
            args[arg['name']] = args_namespace[arg['name_new']]
            setattr(args_dataclass, arg['name'], args_namespace[arg['name_new']])

    return args_dataclass

def parse_args_into_dataclasses(
    dataclass
):
    parser = argparse.ArgumentParser()
    branches, fields = get_dataclass_fields(dataclass)
    for name, ftype, fdefault in fields:
        # print(typing.get_origin(type))
        # if typing.get_origin()
        if typing.get_origin(ftype) == typing.Union:
            literal_args = typing.get_args(typing.get_args(ftype)[0])
            arg_type = type(literal_args[0])
            parser.add_argument(f'--{name}', type=arg_type, choices=literal_args, default=fdefault)
        elif typing.get_origin(ftype) == Literal:
            literal_args = typing.get_args(ftype)
            arg_type = type(literal_args[0])
            parser.add_argument(f'--{name}', type=arg_type, choices=literal_args, default=fdefault)
        else:
            parser.add_argument(f'--{name}', type=ftype, default=fdefault)
    
    args_namespace = parser.parse_args()
    args_dataclass = parse_args_into_dataclass(branches, vars(args_namespace))

    return args_dataclass


if __name__ == '__main__':

    config = Base_Config
