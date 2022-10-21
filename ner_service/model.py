import os

import numpy as np
import py_vncorenlp
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from transformers import TFRobertaForTokenClassification as Classifier

from ner_service import constants
from ner_service.crf import CRFModel

PADDING_TAG = "PAD"
CURRENT_DIR = os.getcwd()


def build_model(model_name: str, num_labels: int, num_vocab: int = None):
    if model_name == constants.MODEL_PHOBERT_BASE:
        return Classifier.from_pretrained("vinai/phobert-base",
                                          trainable=False,
                                          num_labels=num_labels,
                                          output_hidden_states=False,
                                          output_attentions=False)
    elif model_name == constants.MODEL_PHOBERT_LARGE:
        return Classifier.from_pretrained("vinai/phobert-large",
                                          trainable=False,
                                          num_labels=num_labels,
                                          output_hidden_states=False,
                                          output_attentions=False)
    elif model_name == constants.MODEL_BILSTM:
        input = tf.keras.Input(shape=(constants.MAX_TOKEN_LEN,))
        emb = tf.keras.layers.Embedding(
            input_dim=num_vocab,
            output_dim=constants.BILSTM_EMB_DIM,
            input_length=constants.MAX_TOKEN_LEN
        )(input)
        dropout = tf.keras.layers.Dropout(constants.DROP_OUT)(emb)
        bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(constants.BILSTM_NUM_UNITS,
                                 return_sequences=True,
                                 recurrent_dropout=constants.DROP_OUT)
        )(dropout)
        out = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_labels, activation="softmax")
        )(bilstm)
        return tf.keras.Model(input, out)
    elif model_name == constants.MODEL_BILSTM_CRF:
        input = tf.keras.Input(shape=(constants.MAX_TOKEN_LEN,))
        emb = tf.keras.layers.Embedding(
            input_dim=num_vocab,
            output_dim=constants.BILSTM_EMB_DIM,
            input_length=constants.MAX_TOKEN_LEN
        )(input)
        bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(constants.BILSTM_NUM_UNITS,
                                 return_sequences=True,
                                 recurrent_dropout=constants.DROP_OUT)
        )(emb)
        dense = tf.keras.layers.Dense(num_labels, activation="relu")(bilstm)
        base = tf.keras.Model(inputs=input, outputs=dense)
        return CRFModel(base, num_labels)
    elif model_name == constants.MODEL_PHOBERT_BASE_BILSTM:
        input = tf.keras.Input(shape=(constants.MAX_TOKEN_LEN,))
        bert_model = TFAutoModel.from_pretrained("vinai/phobert-base",
                                                 trainable=False)
        emb = bert_model.roberta(input)["last_hidden_state"]
        dropout = tf.keras.layers.Dropout(constants.DROP_OUT)(emb)
        bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(constants.BILSTM_NUM_UNITS,
                                 return_sequences=True,
                                 recurrent_dropout=constants.DROP_OUT)
        )(dropout)
        out = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_labels, activation="softmax")
        )(bilstm)
        return tf.keras.Model(input, out)
    elif model_name == constants.MODEL_PHOBERT_LARGE_BILSTM:
        input = tf.keras.Input(shape=(constants.MAX_TOKEN_LEN,))
        bert_model = TFAutoModel.from_pretrained("vinai/phobert-large",
                                                 trainable=False)
        emb = bert_model.roberta(input)["last_hidden_state"]
        dropout = tf.keras.layers.Dropout(constants.DROP_OUT)(emb)
        bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(constants.BILSTM_NUM_UNITS,
                                 return_sequences=True,
                                 recurrent_dropout=constants.DROP_OUT)
        )(dropout)
        out = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_labels, activation="softmax")
        )(bilstm)
        return tf.keras.Model(input, out)
    elif model_name == constants.MODEL_PHOBERT_BASE_BILSTM_CRF:
        input = tf.keras.Input(shape=(constants.MAX_TOKEN_LEN,))
        bert_model = TFAutoModel.from_pretrained("vinai/phobert-base",
                                                 trainable=False)
        emb = bert_model.roberta(input)["last_hidden_state"]
        bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(constants.BILSTM_NUM_UNITS,
                                 return_sequences=True,
                                 recurrent_dropout=constants.DROP_OUT)
        )(emb)
        dense = tf.keras.layers.Dense(num_labels, activation="relu")(bilstm)
        base = tf.keras.Model(inputs=input, outputs=dense)
        return CRFModel(base, num_labels)
    elif model_name == constants.MODEL_PHOBERT_LARGE_BILSTM_CRF:
        input = tf.keras.Input(shape=(constants.MAX_TOKEN_LEN,))
        bert_model = TFAutoModel.from_pretrained("vinai/phobert-large",
                                                 trainable=False)
        emb = bert_model.roberta(input)["last_hidden_state"]
        bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(constants.BILSTM_NUM_UNITS,
                                 return_sequences=True,
                                 recurrent_dropout=constants.DROP_OUT)
        )(emb)
        dense = tf.keras.layers.Dense(num_labels, activation="relu")(bilstm)
        base = tf.keras.Model(inputs=input, outputs=dense)
        return CRFModel(base, num_labels)


class NERModel(object):
    def __init__(self) -> None:
        with open('resource/tags.txt', 'r') as f:
            self.__tags = [line.rstrip('\n') for line in f]

        self.load_annotator()
        self.load_tokenizer()

        self.load_bilstm_model()
        # self.load_bilstm_crf_model()
        # self.load_phobert_base_model()
        # self.load_phobert_large_model()
        # self.load_phobert_base_bilstm_model()
        # self.load_phobert_large_bilstm_model()
        self.load_phobert_base_bilstm_crf_model()
        # self.load_phobert_large_bilstm_crf_model()

    def load_annotator(self):
        vncorenlp_dir = os.path.join(CURRENT_DIR, "resource/vncorenlp")
        if not os.path.exists(vncorenlp_dir):
            py_vncorenlp.download_model(save_dir=vncorenlp_dir)
        self.__annotator = py_vncorenlp.VnCoreNLP(annotators=["wseg"],
                                                  save_dir=vncorenlp_dir)
        os.chdir(CURRENT_DIR)

    def load_tokenizer(self):
        self.__base_tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-base", use_fast=False
        )
        self.__large_tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-large", use_fast=False
        )

    def load_bilstm_model(self):
        self.__bilstm_model = tf.keras.models.load_model(
            filepath=constants.RESOURCE_BILSTM
        )

    def load_bilstm_crf_model(self):
        self.__bilstm_crf_model = tf.keras.models.load_model(
            filepath=constants.RESOURCE_BILSTM_CRF
        )

    def load_phobert_base_model(self):
        self.__phobert_base_model = build_model(
            constants.MODEL_PHOBERT_BASE, len(self.__tags))
        self.__phobert_base_model.load_weights(
            constants.RESOURCE_PHOBERT_BASE
        )
        self.__phobert_base_model.compile(
            metrics=["accuracy"]
        )

    def load_phobert_large_model(self):
        self.__phobert_large_model = build_model(
            constants.MODEL_PHOBERT_LARGE, len(self.__tags))
        self.__phobert_large_model.load_weights(
            constants.RESOURCE_PHOBERT_LARGE
        )
        self.__phobert_large_model.compile(
            metrics=["accuracy"]
        )

    def load_phobert_base_bilstm_model(self):
        self.__phobert_base_bilstm_model = tf.keras.models.load_model(
            filepath=constants.RESOURCE_PHOBERT_BASE_BILSTM
        )

    def load_phobert_large_bilstm_model(self):
        self.__phobert_large_bilstm_model = tf.keras.models.load_model(
            filepath=constants.RESOURCE_PHOBERT_LARGE_BILSTM
        )

    def load_phobert_base_bilstm_crf_model(self):
        self.__phobert_base_bilstm_crf_model = tf.keras.models.load_model(
            filepath=constants.RESOURCE_PHOBERT_BASE_BILSTM_CRF
        )

    def load_phobert_large_bilstm_crf_model(self):
        self.__phobert_large_bilstm_crf_model = tf.keras.models.load_model(
            filepath=constants.RESOURCE_PHOBERT_LARGE_BILSTM_CRF
        )

    def get_tokenizer(self, model_name: str):
        if model_name.find("large") >= 0:
            return self.__large_tokenizer
        else:
            return self.__base_tokenizer

    def get_model(self, model_name: str):
        if model_name == constants.MODEL_BILSTM:
            return self.__bilstm_model
        elif model_name == constants.MODEL_BILSTM_CRF:
            return self.__bilstm_crf_model
        elif model_name == constants.MODEL_PHOBERT_BASE:
            return self.__phobert_base_model
        elif model_name == constants.MODEL_PHOBERT_LARGE:
            return self.__phobert_large_model
        elif model_name == constants.MODEL_PHOBERT_BASE_BILSTM:
            return self.__phobert_base_bilstm_model
        elif model_name == constants.MODEL_PHOBERT_LARGE_BILSTM:
            return self.__phobert_large_bilstm_model
        elif model_name == constants.MODEL_PHOBERT_BASE_BILSTM_CRF:
            return self.__phobert_base_bilstm_crf_model
        elif model_name == constants.MODEL_PHOBERT_LARGE_BILSTM_CRF:
            return self.__phobert_large_bilstm_crf_model

    def __split_array(self, arr, max_len):
        if len(arr) <= max_len:
            return [arr.copy()]
        idx = max_len
        # 4 = . ; 5 = ,
        while arr[idx] != 4 and arr[idx] != 5 and idx > 0:
            idx -= 1
        if idx == 0:
            idx = max_len
        results = [arr[:idx].copy()]
        results.extend(self.__split_array(arr[idx:].copy(), max_len))
        return results

    def predict_sentence(self, sentence, model_name):
        tokenizer = self.get_tokenizer(model_name)
        model = self.get_model(model_name)
        segmented_text = ' '.join(self.__annotator.word_segment(sentence))
        x = tokenizer.encode(segmented_text, add_special_tokens=True)
        xs = self.__split_array(x, constants.MAX_TOKEN_LEN)
        xs = tf.keras.preprocessing.sequence.pad_sequences(
            xs,
            maxlen=constants.MAX_TOKEN_LEN,
            dtype="long",
            truncating="post",
            padding="post",
            value=tokenizer.pad_token_id,
        )
        ys = model.predict(xs)
        if model_name.endswith("crf"):
            ys = ys[0]
        elif model_name.endswith("bilstm"):
            ys = np.argmax(ys, axis=-1)
        else:
            ys = np.argmax(ys[0], axis=-1)

        new_tokens, new_tags = [], []
        for idx, x in enumerate(xs):
            label_indices = ys[idx]
            tokens = self.__base_tokenizer.convert_ids_to_tokens(x)

            for token, label_idx in zip(tokens, label_indices):
                if token == "<s>" or token == "</s>" or token == "<pad>":
                    continue
                tag = self.__tags[label_idx]
                if tag == PADDING_TAG:
                    tag = "O"
                new_tags.append(tag)
                new_tokens.append(token)

        new_tokens, new_tags = self.__convert_subwords_to_text(
            new_tokens, new_tags)
        new_tokens = self.__fix_unknown_tokens(new_tokens, segmented_text)
        return self.__generate_content_label(new_tokens, new_tags)

    def __convert_subwords_to_text(self, tokens, tags):
        idx = 0
        tks, ts = [], []
        while idx < len(tags):
            current_tag = tags[idx]
            if current_tag == PADDING_TAG:
                current_tag = "O"

            current_token = tokens[idx]
            updated = True
            while current_token.endswith("@@"):
                idx += 1
                if idx < len(tags):
                    if tokens[idx] != "," and tokens[idx] != ".":
                        current_token += " " + tokens[idx]
                        current_token = current_token.replace("@@ ", "")
                        if current_tag == "O":
                            current_tag = tags[idx]
                    else:
                        updated = False
                        current_token = current_token.replace("@@", "")
                        break
                else:
                    updated = False
                    current_token = current_token.replace("@@", "")
                    break
            tks.append(current_token)
            ts.append(current_tag)
            if updated:
                idx += 1

        return tks, ts

    def __generate_content_label(self, tokens, tags):
        contents, labels = [], []
        idx = 0
        while idx < len(tags):
            ts = []
            current_tag = tags[idx]
            current_token = tokens[idx]

            if current_tag == "O":
                current_label = current_tag
                ts.append(current_token)
                labels.append(current_label)

                idx += 1
                while idx < len(tags):
                    next_tag = tags[idx]
                    if next_tag != "O":
                        break
                    next_token = tokens[idx]
                    ts.append(next_token)
                    idx += 1

                content = ' '.join(ts)
                content = content.replace(" _ ", " ")
                content = content.replace("_", " ")
                contents.append(content)
                continue

            if current_tag.startswith('B-') or current_tag.startswith('I-'):
                current_label = current_tag[2:]
                ts.append(current_token)
                labels.append(current_label)

                idx += 1
                while idx < len(tags):
                    next_tag = tags[idx]
                    next_label = next_tag[2:]
                    if next_label != current_label:
                        break
                    next_token = tokens[idx]
                    ts.append(next_token)
                    idx += 1

                content = ' '.join(ts)
                content = content.replace(" _ ", " ")
                content = content.replace("_", " ")
                contents.append(content)
                continue

        return contents, labels

    def __fix_unknown_tokens(self, tokens, segmented_text):
        current_pos = 0
        for idx, token in enumerate(tokens):
            if token == "<unk>" and idx < len(tokens)-1:
                next_token = tokens[idx+1]
                pos = segmented_text.find(next_token, current_pos)
                if pos >= 0:
                    tokens[idx] = segmented_text[current_pos:pos-1]
            current_pos += len(tokens[idx]) + 1
        return tokens
