# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os

from ...file_utils import is_tf_available
from .utils import DataProcessor, InputExample, InputFeatures


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def review_sentiment_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = review_sentiment_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = review_sentiment_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            try:
                label = label_map[example.label]
            except:
                import pdb
                print(example.label, label_map)
                pdb.set_trace()
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features



class ReviewSentimentProcessor(DataProcessor):
    """Processor for the ReviewSentiment multi domain data set used in
    https://www.ijcai.org/Proceedings/2019/0681.pdf
    https://www.aclweb.org/anthology/P17-1001.pdf."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_mtl_sent_data(data_dir+'train')[:-200], "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_mtl_sent_data(data_dir+'train')[-200:], "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_mtl_sent_data(data_dir+'test'), "test")

    # def _read_one_file(self, input_file, label, lines):
    #     # import pdb
    #     with open(input_file, "r",) as f:
    #         line = {}
    #         for text_line in f:
    #             text_line = text_line.strip()
    #             # print("text_line: ", text_line, "line: ", line, " lines: ", lines, flush=True)
    #             if '<review>' in text_line:
    #                 line = {}
    #                 tag = None
    #                 value = ''
    #             elif '</review>' in text_line:
    #                 if len(line)>0 and 'review_text' in line.keys():
    #                     line.update({'label': label})
    #                     lines.append(line)
    #                     # print("line: ", line, flush=True)
    #
    #             elif text_line.startswith("</"):
    #                 if tag: line.update({tag:value})
    #                 tag = None
    #             elif text_line.startswith("<"):
    #                 tag = text_line[1:-1]
    #                 value = ''
    #             else:
    #                 value += text_line
    #     # pdb.set_trace()

    # def _read_xml(self, input_dir):
    #     """Reads a tab separated value file."""
    #     lines = []
    #     neg_input_file = os.path.join(input_dir, 'negative.review')
    #     pos_input_file = os.path.join(input_dir, 'positive.review')
    #     self._read_one_file(neg_input_file, "0", lines)
    #     self._read_one_file(pos_input_file, "1", lines)
    #     return  lines
    def _read_mtl_sent_data(self,input_file_name):
        lines = []
        with open(input_file_name, "r", ) as f:
            for line in f:
                lines.append({'label': line[0], "review_text": line[1:].strip()})
        return lines

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['review_text']
            label = line['label']
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

review_sentiment_tasks_num_labels = {
    "books": 2,
    'kitchen_housewares': 2,
    'dvd': 2,
    'electronics': 2,
    'apparel': 2,
    'camera_photo': 2,
    'baby': 2,
    'health_personal_care': 2,
    'magazines': 2,
    'MR': 2,
    # 'music': 2,
    # 'imdb': 2,
    'software': 2,
    'video': 2,
    'toys_games': 2,
    'sports_outdoors': 2,
}

review_sentiment_processors = {

    "books": ReviewSentimentProcessor,
    'kitchen_housewares': ReviewSentimentProcessor,
    'dvd': ReviewSentimentProcessor,
    'electronics': ReviewSentimentProcessor,
    'apparel': ReviewSentimentProcessor,
    'camera_photo': ReviewSentimentProcessor,
    'baby': ReviewSentimentProcessor,
    'health_personal_care': ReviewSentimentProcessor,
    'magazines': ReviewSentimentProcessor,
    'MR': ReviewSentimentProcessor,
    # 'music': ReviewSentimentProcessor,
    # 'imdb': ReviewSentimentProcessor,
    'software': ReviewSentimentProcessor,
    'video': ReviewSentimentProcessor,
    'toys_games': ReviewSentimentProcessor,
    'sports_outdoors': ReviewSentimentProcessor,

}

review_sentiment_output_modes = {
    "books": "classification",
    'kitchen_housewares':"classification",
    'dvd':"classification",
    'electronics':"classification",
    'apparel':"classification",
    'camera_photo':"classification",
    'baby':"classification",
    'health_personal_care':"classification",
    'magazines':"classification",
    'MR':"classification",
    # 'music':"classification",
    # 'imdb':"classification",
    'software':"classification",
    'video':"classification",
    'toys_games':"classification",
    'sports_outdoors':"classification",
}
