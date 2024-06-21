# Copyright 2023 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel, base_architecture, \
    s2t_transformer_m, s2t_transformer_s
from examples.speech_to_text.models.multi_task import MultiTaskClassifierModel
from fairseq.models import register_model, register_model_architecture


@register_model('multitask_s2t_transformer')
class MultitaskConvolutionalTransformer(MultiTaskClassifierModel):
    @staticmethod
    def add_args(parser):
        S2TTransformerModel.add_args(parser)
        MultiTaskClassifierModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        base_model = S2TTransformerModel.build_model(args, task)
        return cls.build_with_classifier(base_model, args, task)


@register_model_architecture('multitask_s2t_transformer', 'multitask_s2t_transformer')
def base_multitask_architecture(args):
    base_architecture(args)


@register_model_architecture('multitask_s2t_transformer', 'multitask_s2t_transformer_s')
def multitask_s2t_transformer_s(args):
    s2t_transformer_s(args)


@register_model_architecture('multitask_s2t_transformer', 'multitask_s2t_transformer_m')
def multitask_s2t_transformer_m(args):
    s2t_transformer_m(args)
