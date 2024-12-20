import os
import json
from transformers.generation.configuration_utils import GenerationConfig
import typing
from typing import Literal, Optional
from types import SimpleNamespace
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

    def to_dict(self):

        return asdict(self)
    
    def from_dict(self, input_dict: dict):
        
        for name, value in input_dict.items():
            if hasattr(self, name):
                if isinstance(value, dict):
                    sub_dataclass = getattr(self, name)
                    sub_dataclass.from_dict(value)
                else:
                    setattr(self, name, value)
            else:
                raise KeyError(f'Invalid key \'{name}\' in dataclass \'{type(self)}\'.')

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
            if field.name.endswith('__args_namespace__'):
                continue
            name_new = '.'.join([prefix, field.name]) if prefix is not None else field.name
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
                    # field.default if default is None else getattr(default, field.name)
                    getattr(default, field.name)
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
        branches, fields = self.get_dataclass_fields(
            dataclass = type(self),
            default = self
        )

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

        return args_dataclass

    def get_args_info(self):

        _, fields = self.get_dataclass_fields(
            dataclass = type(self),
            default = self
        )

        args_info_str = 'Args Info:\n'
        for (name_parse, ftype, value) in fields:
            args_info_str += f'--{name_parse} {value}\n'

        return args_info_str
    
    def to_json(self, path):

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    def from_json(self, path):

        with open(path, 'r') as f:
            self.from_dict(json.load(f))
        

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
    eos_token_id = 50256,
    num_beams = 4,
    repetition_penalty = 0.5,
    no_repeat_ngram_size = 5
)

@dataclass
class Accelertor_Config(Base_Config):

    log_with: Optional[Literal['wandb', 'tensorboard']] = None
    gradient_accumulation_steps: int = 1

@dataclass
class Trainer_Config(Base_Config):

    task_name: str = None
    n_save_step: int = None
    output_dir: str = os.path.join('.', 'output')


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

    config = Base_Config()
